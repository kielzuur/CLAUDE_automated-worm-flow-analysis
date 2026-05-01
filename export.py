import re
import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')


def make_export_filename(base_name: str, param: str, operation: str, suffix: str) -> str:
    safe_base = re.sub(r'[^\w\-]', '_', base_name)[:30].strip('_')
    safe_param = re.sub(r'\s+', '_', param)
    safe_op = re.sub(r'\s+', '_', operation)
    return f"{safe_base}_{safe_param}_{safe_op}_{suffix}.csv"
