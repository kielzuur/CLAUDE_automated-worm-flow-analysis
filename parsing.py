import io
import pandas as pd

SPACER_OPTIONS = {
    "No spacers (use all wells)":             "none",
    "Even columns are spacers (2, 4, 6…)":    "even",
    "Odd columns are spacers (1, 3, 5…)":     "odd",
    "Two filled, one spacer (cols 3, 6, 9…)": "two_one",
}

NON_ANALYTIC = {
    'Id', 'Time', 'Plate', 'Row', 'Column',
    'Source well', 'Clog', 'Sorted status', 'In Regions',
}

PREFERRED_ORDER = [
    'Green', 'Blue', 'Yellow', 'Red',
    'TOF', 'Extinction',
    'PH Green', 'PH Blue', 'PH Yellow', 'PH Red', 'PH Extinction',
    'PW Green', 'PW Blue', 'PW Yellow', 'PW Red', 'PW Extinction',
    'PC Green', 'PC Blue', 'PC Yellow', 'PC Red', 'PC Extinction',
]


def parse_uploaded_file(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    try:
        text = file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        text = file_bytes.decode('latin-1')

    lines = text.splitlines(keepends=True)
    if not lines:
        raise ValueError("File is empty.")

    header = lines[0]
    if not header.startswith('Id\t'):
        raise ValueError(
            "File does not appear to be a flow cytometry export "
            "(expected header starting with 'Id\\t')."
        )

    kept = [header]
    for line in lines[1:]:
        first_field = line.split('\t', 1)[0].strip()
        if first_field.isdigit():
            kept.append(line)

    if len(kept) < 2:
        raise ValueError("No data rows found after stripping the instrument footer.")

    df = pd.read_csv(
        io.StringIO(''.join(kept)),
        sep='\t',
        na_values=['n/a'],
        low_memory=False,
    )

    unnamed = [c for c in df.columns if str(c).startswith('Unnamed:')]
    if unnamed:
        df = df.drop(columns=unnamed)

    if 'Source well' not in df.columns:
        raise ValueError("No 'Source well' column found in file.")

    df['Source well'] = df['Source well'].astype(str).str.strip()

    return df


def get_analyzable_columns(df: pd.DataFrame) -> list:
    candidates = [
        c for c in df.columns
        if c not in NON_ANALYTIC and pd.api.types.is_numeric_dtype(df[c])
    ]
    ordered = [p for p in PREFERRED_ORDER if p in candidates]
    remaining = sorted(c for c in candidates if c not in ordered)
    return ordered + remaining


def sort_wells(wells) -> list:
    result = []
    for w in wells:
        w = str(w).strip()
        try:
            key = (w[0].upper(), int(w[1:]))
        except (IndexError, ValueError):
            key = (w, 0)
        result.append((key, w))
    return [w for _, w in sorted(result)]


def _bare_well(well: str) -> str:
    """Strip file-prefix from combined-mode labels like 'stem/A1' → 'A1'."""
    return well.split('/')[-1].strip()


def filter_spacer_wells(wells: list, pattern: str) -> list:
    """Return wells that are not spacers under the chosen plate layout pattern."""
    if pattern == "none":
        return wells
    result = []
    for w in wells:
        bare = _bare_well(w)
        try:
            col_num = int(bare[1:])
        except (IndexError, ValueError):
            result.append(w)
            continue
        if pattern == "even" and col_num % 2 == 0:
            continue
        if pattern == "odd" and col_num % 2 == 1:
            continue
        if pattern == "two_one" and col_num % 3 == 0:
            continue
        result.append(w)
    return result
