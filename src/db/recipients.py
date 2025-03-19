import os
from typing import List, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError

from .data_models import prod_user_tbl, test_user_tbl


def retrieve_recipients(test_mode: bool) -> Optional[List[str]]:
    table = test_user_tbl if test_mode else prod_user_tbl
    print(f"Fetching users from `{table.name}` table")
    statement = select(table).distinct()
    engine = create_engine(os.environ["DATABASE_URL"])
    try:
        with engine.connect() as conn:
            with conn.begin():
                res = conn.execute(statement)
                return [i.email for i in res.fetchall()]

    except SQLAlchemyError as e:
        print(f"Error connecting to Cockroach DB:\n{e}")
        raise e
    finally:
        engine.dispose()
