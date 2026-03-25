# CSV Cut Demo (Render + Python)

This is a minimal “render backend” demo:
1. Accept an uploaded CSV
2. Use Python to cut it down to the first `n` data rows (header always kept)
3. Return the output CSV

## Local run

```bash
pip install -r requirements.txt
python server.py
```

## Local test

Replace `County General Data.csv` with your CSV filename if needed.

```bash
curl -X POST http://localhost:5000/cut \
  -F "file=@\"County General Data.csv\"" \
  -F "n=5" \
  -o forecast_output.csv
```

## API

- `POST /cut`
  - form-data field `file`: CSV file
  - form-data field `n` (optional): number of data rows to keep (default `5`)

- returns `text/csv` attachment

## Where to plug your ML later

You can replace the body of `cut()` in `server.py` with a call to your ML code,
and still return the produced CSV the same way.

