from sqlalchemy import Column, MetaData, String, Table

test_user_tbl = Table(
    "test",
    MetaData(),
    Column("email", String),
)

prod_user_tbl = Table(
    "prod",
    MetaData(),
    Column("email", String),
)
