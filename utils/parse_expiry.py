import polars as pl
from datetime import date


class ParseExpiry:
    def parse_expiry_expr(self, col_name: str = "Expiry_Date") -> pl.Expr:
        return pl.coalesce([
            pl.col(col_name).cast(pl.Date, strict=False),
            pl.col(col_name).str.strptime(pl.Date, format=None, strict=False),
            pl.col(col_name).str.to_date(format=None, strict=False)
        ])


    def parse_expiry_with_excel_serial(self, col_name: str = "Expiry_Date") -> pl.Expr:
        excel_origin = pl.lit(date(1899, 12, 30))  # origen estándar Excel (con el bug de 1900)
        excel_serial_to_date = (
            excel_origin
            + pl.duration(days=pl.col(col_name).cast(pl.Int64, strict=False))
        )
        return pl.coalesce([
            pl.col(col_name).cast(pl.Date, strict=False),
            pl.col(col_name).str.strptime(pl.Date, format=None, strict=False),
            pl.col(col_name).str.to_date(format=None, strict=False),
            excel_serial_to_date
        ])
