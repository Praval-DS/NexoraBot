import json
import re
import sqlite3
import pandas as pd
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from src.config.index import appConfig
from src.services.llm import openAI


class SmartDataAgent:
    def __init__(self, file_paths: List[str], schema_json_path: str, model_name: str = "gpt-4o"):
        self.file_paths = file_paths
        self.schema_json_path = schema_json_path
        self.model_name = model_name

        self.llm = openAI["chat_llm"]
        self.mini_llm = openAI["mini_llm"]

        with open(schema_json_path, 'r') as f:
            self.schema = json.load(f)

        self.data_profiles: Dict[str, Dict] = {}

        self.conn = sqlite3.connect(":memory:")
        self._load_data_to_sqlite()

    def _profile_dataframe(self, df: pd.DataFrame) -> Dict:
        profile = {}
        for col in df.columns:
            col_data = df[col].dropna()
            entry = {
                "dtype": str(df[col].dtype),
                "null_pct": round(df[col].isna().mean() * 100, 1),
                "unique_count": int(df[col].nunique()),
                "row_count": len(df),
                "sample_vals": col_data.head(5).tolist(),
            }
            if pd.api.types.is_numeric_dtype(df[col]) and len(col_data) > 0:
                entry.update({
                    "min":    round(float(col_data.min()),    4),
                    "max":    round(float(col_data.max()),    4),
                    "mean":   round(float(col_data.mean()),   4),
                    "std":    round(float(col_data.std()),    4),
                    "median": round(float(col_data.median()), 4),
                })
            profile[col] = entry
        return profile

    def _load_data_to_sqlite(self):
        for path in self.file_paths:
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            elif path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(path)
            else:
                continue

            simple_filename = path.split('/')[-1].split('\\')[-1].split('.')[0].lower()

            matched_table_name = None
            if self.schema and "tables" in self.schema:
                for table_def in self.schema["tables"]:
                    schema_name = table_def["table_name"].lower()
                    if schema_name in simple_filename or simple_filename in schema_name:
                        matched_table_name = schema_name
                        break

            table_name = matched_table_name if matched_table_name else simple_filename

            print(f"DEBUG: Loading file {path} into table '{table_name}'")
            df.to_sql(table_name, self.conn, index=False, if_exists='replace')

            self.data_profiles[table_name] = self._profile_dataframe(df)
            print(f"DEBUG: Profiled '{table_name}': {list(self.data_profiles[table_name].keys())}")

        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"DEBUG: Tables in SQLite: {tables}")

    def _get_loaded_tables(self) -> list:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()]

    def filter_schema(self, user_query: str) -> Dict:
        system_prompt = """You are a strictly logical Data Architect.
        Given a user query and a database schema, return a JSON object containing ONLY
        the tables and columns strictly required to generate a SQL query for the answer.

        IMPORTANT:
        - If the user asks about "all tables", "the whole schema", or "database statistics", return ALL tables and their primary keys/relevant columns.
        - Do NOT iterate or explain. Just output the JSON.

        Output format:
        {{
            "tables": [
                {{
                    "table_name": "name",
                    "definition": "description of table",
                    "columns": [
                        {{
                            "name": "col_name",
                            "definition": "description",
                            "data_type": "type",
                            "key_type": "PK/FK/None",
                            "example": "example_value"
                        }}
                    ]
                }}
            ]
        }}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Schema: {schema}\n\nQuery: {query}")
        ])

        chain = prompt | self.llm | JsonOutputParser()
        return chain.invoke({"schema": json.dumps(self.schema), "query": user_query})

    def generate_sql(self, user_query: str, filtered_schema: Dict) -> str:
        reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a senior data analyst reviewing a schema before writing SQL.
Think step by step:
1. Which table(s) are required to answer this question?
2. Which exact column names (from the schema) are needed?
3. What aggregation, filter, GROUP BY, or JOIN is required?
4. Are there any edge cases to handle (NULLs, type casting, case sensitivity)?

Output ONLY your reasoning as plain text. Do NOT write SQL yet."""),
            ("user", "Schema: {schema}\n\nData profile (actual stats): {profile}\n\nQuestion: {query}")
        ])

        profile_summary = {
            tbl: {col: {k: v for k, v in stats.items() if k != "sample_vals"}
                  for col, stats in cols.items()}
            for tbl, cols in self.data_profiles.items()
        }

        reasoning = (reasoning_prompt | self.llm | StrOutputParser()).invoke({
            "schema":  json.dumps(filtered_schema),
            "profile": json.dumps(profile_summary),
            "query":   user_query,
        })
        print(f"DEBUG: CoT Reasoning:\n{reasoning}")

        sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL developer for SQLite.
Use the analyst's reasoning to write a precise SQL query.

STRICT RULES:
1. Return ONLY the raw SQL string — no markdown, no explanation.
2. Use ONLY column names that exist verbatim in the provided schema.
3. For row counts across all tables use UNION ALL.
4. If the question is unanswerable from the schema, return exactly: NO_SQL_POSSIBLE"""),
            ("user", "Schema: {schema}\n\nAnalyst reasoning: {reasoning}\n\nQuestion: {query}")
        ])

        sql = (sql_prompt | self.llm | StrOutputParser()).invoke({
            "schema":    json.dumps(filtered_schema),
            "reasoning": reasoning,
            "query":     user_query,
        })
        sql = sql.replace("```sql", "").replace("```", "").strip()
        print(f"DEBUG: Raw generated SQL: {sql}")

        if "NO_SQL_POSSIBLE" not in sql:
            sql = self._validate_and_fix_sql(sql, user_query, filtered_schema)

        return sql

    def _validate_and_fix_sql(self, sql: str, user_query: str, filtered_schema: Dict) -> str:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables_in_db = {row[0].lower() for row in cursor.fetchall()}

        all_valid_columns: set = set()
        for table in tables_in_db:
            cursor.execute(f"PRAGMA table_info({table});")
            all_valid_columns.update(row[1].lower() for row in cursor.fetchall())

        SQL_KEYWORDS = {
            "select", "from", "where", "group", "by", "order", "having",
            "join", "on", "and", "or", "not", "in", "is", "null", "as",
            "distinct", "limit", "count", "sum", "avg", "min", "max",
            "union", "all", "left", "right", "inner", "outer", "cross",
            "case", "when", "then", "else", "end", "between", "like",
            "exists", "with", "over", "partition", "desc", "asc", "cast",
            "coalesce", "ifnull", "round", "substr", "length", "trim",
            "upper", "lower", "replace", "strftime", "date", "rowid",
            # data types that appear inside CAST()
            "integer", "int", "text", "real", "blob", "numeric", "varchar",
            "boolean", "float", "double", "char", "bigint", "smallint",
            # common aliases the LLM generates after AS
            "total", "result", "value", "output", "name",
        }

        # ── Table guard ────────────────────────────────────────────────────────
        sql_lower = sql.lower()
        table_refs = re.findall(r'\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql_lower)
        table_refs += re.findall(r'\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)', sql_lower)
        bad_tables = [t for t in table_refs if t not in tables_in_db and t not in SQL_KEYWORDS]

        if bad_tables:
            print(f"DEBUG: Table guard caught non-existent tables: {bad_tables}")
            loaded = self._get_loaded_tables()

            # Check if the filtered schema has ANY overlap with loaded tables.
            # If not, this is a schema mismatch — retrying SQL won't help.
            schema_table_names = [
                t["table_name"].lower()
                for t in filtered_schema.get("tables", [])
            ]
            overlap = [t for t in schema_table_names if t in [x.lower() for x in loaded]]

            if not overlap:
                mismatch_msg = (
                    f"The uploaded file contains the table(s) {loaded}, but your question "
                    f"was answered using a schema that references {bad_tables}. "
                    f"Make sure the correct schema is linked to this project."
                )
                print(f"DEBUG: Schema mismatch — {mismatch_msg}")
                return f"SCHEMA_MISMATCH: {mismatch_msg}"

            fix_prompt = ChatPromptTemplate.from_messages([
                ("system", "Fix the SQL. These tables don't exist: {bad_tables}. "
                           "Available tables: {available_tables}. "
                           "Return ONLY corrected raw SQL."),
                ("user", "Schema: {schema}\n\nBroken SQL: {sql}\n\nOriginal question: {query}")
            ])
            sql = (fix_prompt | self.llm | StrOutputParser()).invoke({
                "bad_tables":       str(bad_tables),
                "available_tables": str(loaded),
                "schema":           json.dumps(filtered_schema),
                "sql":              sql,
                "query":            user_query,
            })
            sql = sql.replace("```sql", "").replace("```", "").strip()
            print(f"DEBUG: Table-fixed SQL: {sql}")

        # ── Column guard ───────────────────────────────────────────────────────
        # Strip AS aliases so "AS total_revenue" doesn't flag "total_revenue"
        # as a missing column reference.
        sql_no_aliases = re.sub(r'\bAS\s+[a-zA-Z_][a-zA-Z0-9_]*', '', sql, flags=re.IGNORECASE)
        tokens = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', sql_no_aliases.lower()))
        suspect = tokens - SQL_KEYWORDS - tables_in_db - {"*"}
        bad_cols = [c for c in suspect if c not in all_valid_columns]

        if not bad_cols:
            print("DEBUG: Column guard passed — all identifiers valid.")
            return sql

        print(f"DEBUG: Column guard caught bad columns: {bad_cols}. Retrying...")

        fix_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL debugger.
The SQL below references columns that do not exist in the database.
Fix the query using ONLY the columns available in the schema.
Return ONLY the corrected raw SQL. No markdown, no explanation."""),
            ("user",
             "Available schema: {schema}\n\n"
             "Non-existent columns found: {bad_cols}\n\n"
             "Broken SQL: {sql}\n\n"
             "Original question: {query}")
        ])

        fixed_sql = (fix_prompt | self.llm | StrOutputParser()).invoke({
            "schema":   json.dumps(filtered_schema),
            "bad_cols": str(bad_cols),
            "sql":      sql,
            "query":    user_query,
        })
        fixed_sql = fixed_sql.replace("```sql", "").replace("```", "").strip()
        print(f"DEBUG: Fixed SQL: {fixed_sql}")
        return fixed_sql

    def execute_and_answer(self, user_query: str):
        # FIX 2 — sql initialized before try so both except blocks can reference it
        # without a "cannot access local variable" error if generate_sql() throws
        sql = ""
        try:
            print(f"DEBUG: User Query: '{user_query}'")

            filtered_schema = self.filter_schema(user_query)
            print(f"DEBUG: Filtered Schema: {json.dumps(filtered_schema, indent=2)}")

            sql = self.generate_sql(user_query, filtered_schema)
            print(f"DEBUG: Final SQL: {sql}")

            if "NO_SQL_POSSIBLE" in sql:
                return {
                    "answer": "I could not generate a valid SQL query for this question. "
                              "The data needed may not be present in the uploaded files.",
                    "sql": sql,
                    "data": [],
                }

            # Surface schema mismatch as a clean user-facing message
            if sql.startswith("SCHEMA_MISMATCH:"):
                return {
                    "answer": sql.replace("SCHEMA_MISMATCH: ", ""),
                    "sql": "",
                    "data": [],
                }

            cursor = self.conn.cursor()
            cursor.execute(sql)
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                data_result = [dict(zip(columns, row)) for row in results]
            else:
                data_result = []

            text_table = ""
            if data_result:
                headers = list(data_result[0].keys())
                lines = [
                    " | ".join(headers),
                    "-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)),
                ]
                for row in data_result:
                    lines.append(" | ".join(str(v) for v in row.values()))
                text_table = "\n".join(lines)

            explanation = self._contextualise_result(user_query, data_result, sql)

            # Combine AI explanation + raw query result so the frontend
            # shows both the plain-English summary and the actual data table.
            answer = explanation
            if text_table:
                answer = f"{explanation}\n\n**Query result:**\n```\n{text_table}\n```"

            return {
                "answer":    answer,
                "sql":       sql,
                "data":      data_result,
                "raw_table": text_table,
            }

        except sqlite3.OperationalError as e:
            err = str(e)
            if "no such table" in err:
                table_name = err.split("no such table:")[-1].strip()
                friendly = (
                    f"The table '{table_name}' doesn't exist in the uploaded data. "
                    f"Available tables: {self._get_loaded_tables()}"
                )
            elif "no such column" in err:
                col_name = err.split("no such column:")[-1].strip()
                friendly = (
                    f"Column '{col_name}' doesn't exist in the table. "
                    f"Try asking 'show me the columns' to see what's available."
                )
            elif "syntax error" in err:
                friendly = (
                    f"The generated SQL had a syntax error: {err}. "
                    f"Try rephrasing your question more specifically."
                )
            else:
                friendly = f"Database error: {err}"
            return {"answer": friendly, "sql": sql, "data": []}

        except Exception as e:
            return {"answer": f"Unexpected error: {str(e)}", "sql": sql, "data": []}

    def _contextualise_result(self, user_query: str, data_result: list, sql: str) -> str:
        if not data_result:
            return (
                "The query returned no results. "
                "The filter may be too strict, or the data may not exist in the uploaded files."
            )

        profile_context: Dict = {}
        for col in data_result[0].keys():
            for table_profile in self.data_profiles.values():
                if col in table_profile and "mean" in table_profile[col]:
                    profile_context[col] = {
                        k: table_profile[col][k]
                        for k in ("mean", "std", "min", "max", "null_pct")
                        if k in table_profile[col]
                    }

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analyst assistant summarising a SQL query result.

Instructions:
1. In 1-2 sentences explain what the result means in plain English relative to the user's question.
2. Anomaly check: if any numeric value in the result is more than 2 standard deviations
   from the column mean (provided in column_stats), add a ⚠️ line flagging it.
   Formula: anomaly if |value - mean| > 2 * std
3. If the result is 0 or empty for a question that likely expects non-zero, flag it as ⚠️ unexpected.
4. Keep the total response under 5 sentences. Do NOT re-state the SQL."""),
            ("user",
             "User question: {query}\n\n"
             "SQL used: {sql}\n\n"
             "Result (first 10 rows): {result}\n\n"
             "Column stats from data profile: {stats}")
        ])

        try:
            explanation = (prompt | self.mini_llm | StrOutputParser()).invoke({
                "query":  user_query,
                "sql":    sql,
                "result": json.dumps(data_result[:10]),
                "stats":  json.dumps(profile_context),
            })
            return explanation.strip()
        except Exception as e:
            print(f"DEBUG: Contextualiser failed: {e}. Falling back to raw table.")
            headers = list(data_result[0].keys())
            lines = [
                " | ".join(headers),
                "-" * (sum(len(h) for h in headers) + 3 * (len(headers) - 1)),
            ]
            for row in data_result[:20]:
                lines.append(" | ".join(str(v) for v in row.values()))
            return "\n".join(lines)


def create_smart_agent(file_paths, schema_path, model="gpt-4o"):
    return SmartDataAgent(file_paths, schema_path, model)