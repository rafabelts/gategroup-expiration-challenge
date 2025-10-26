import polars as pl


def normalize_text_col(col: pl.Expr) -> pl.Expr:
    """ Trim + quita espacios y caracteres invisibles + upper/title segun campo """
    return (
        col.cast(pl.Utf8)
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )
