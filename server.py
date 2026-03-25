import io
import os
import subprocess
import sys
import tempfile

from flask import Flask, jsonify, request, Response
from werkzeug.utils import secure_filename

from cut_csv import cut_first_rows

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index() -> Response:
    # Render often points users at the root URL; provide a friendly landing page.
    return Response(
        "CSV Cut Demo. Use POST /cut with multipart form-data fields: file=<csv> and optional n=<int>.\n",
        mimetype="text/plain",
    )


@app.route("/health", methods=["GET"])
def health() -> Response:
    return Response("ok\n", mimetype="text/plain")


@app.route("/cut", methods=["POST"])
def cut() -> Response:
    """
    Accept multipart upload:
      - file: CSV file
      - n: optional number of data rows to keep (default=5)

    Returns the cut CSV as an attachment.
    """
    if "file" not in request.files:
        return jsonify({"error": "missing form field: file"}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "empty filename"}), 400

    n_raw = request.form.get("n", "").strip()
    try:
        n = int(n_raw) if n_raw else 5
    except ValueError:
        return jsonify({"error": "n must be an integer"}), 400

    # Render will mount us with only the app directory; use temp storage per request.
    filename = secure_filename(uploaded.filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, f"input_{filename}")
        output_path = os.path.join(tmpdir, "output.csv")

        uploaded.save(input_path)
        try:
            cut_first_rows(input_path, output_path, n)
        except Exception as e:
            return jsonify({"error": f"failed to cut csv: {str(e)}"}), 500

        with open(output_path, "rb") as f:
            csv_bytes = f.read()

    headers = {"Content-Disposition": 'attachment; filename="forecast_output.csv"'}
    return Response(csv_bytes, mimetype="text/csv", headers=headers)


@app.route("/rf", methods=["POST"])
def rf() -> Response:
    """
    Accept two uploaded CSVs and run the RandomForest-based forecast script.

    Form fields (multipart):
      - county_csv: uploaded county_general_data.csv
      - zip_csv: uploaded zip_to_county.csv
      - return_year (optional): if set, return {year}_zipcode_predictions_randforest.csv
      - max_counties (optional): limit number of counties to process (default: 10)
      - n_estimators (optional): RandomForest trees (default: 100)
      - max_depth (optional): RandomForest max_depth (default: 6)
      - prediction_years (optional): comma list, e.g. "2024,2025,2026"
    """
    county_file = request.files.get("county_csv")
    zip_file = request.files.get("zip_csv")

    if county_file is None or zip_file is None:
        return jsonify({"error": "missing form fields: county_csv and zip_csv"}), 400
    if not county_file.filename or not zip_file.filename:
        return jsonify({"error": "empty filename for county_csv or zip_csv"}), 400

    return_year_raw = request.form.get("return_year", "").strip()
    return_year = int(return_year_raw) if return_year_raw else None

    def _int_field(name: str, default: int) -> int:
        raw = request.form.get(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            raise ValueError(f"{name} must be an integer")

    try:
        max_counties = _int_field("max_counties", 10)
        n_estimators = _int_field("n_estimators", 100)
        max_depth = _int_field("max_depth", 6)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    prediction_years = request.form.get("prediction_years", "2024,2025,2026").strip()
    if not prediction_years:
        prediction_years = "2024,2025,2026"

    # Script path inside the Render app directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "rf_zipcode_multiyear.py")

    with tempfile.TemporaryDirectory() as tmpdir:
        county_path = os.path.join(tmpdir, "county_general_data.csv")
        zip_path = os.path.join(tmpdir, "zip_to_county.csv")
        output_path = os.path.join(tmpdir, "county_population_forecast.csv")
        zip_output_dir = os.path.join(tmpdir, "zipcode_predictions")

        county_file.save(county_path)
        zip_file.save(zip_path)

        cmd = [
            sys.executable,
            script_path,
            "--input",
            county_path,
            "--zip-map",
            zip_path,
            "--output",
            output_path,
            "--zip-output-dir",
            zip_output_dir,
            "--prediction-years",
            prediction_years,
            "--max-counties",
            str(max_counties),
            "--n-estimators",
            str(n_estimators),
            "--max-depth",
            str(max_depth),
            "--no-plots",
        ]

        proc = subprocess.run(
            cmd,
            cwd=script_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode != 0:
            # Return only tail to keep response size manageable.
            stderr_tail = (proc.stderr or "").strip()[-4000:]
            return (
                jsonify(
                    {
                        "error": "forecast script failed",
                        "stderr_tail": stderr_tail,
                    }
                ),
                500,
            )

        if return_year is None:
            file_path = output_path
            download_name = "county_population_forecast.csv"
        else:
            file_path = os.path.join(zip_output_dir, f"{return_year}_zipcode_predictions_randforest.csv")
            download_name = f"zipcode_predictions_{return_year}.csv"

        if not os.path.exists(file_path):
            return jsonify({"error": f"expected output not found: {os.path.basename(file_path)}"}), 500

        with open(file_path, "rb") as f:
            out_bytes = f.read()

    headers = {"Content-Disposition": f'attachment; filename="{download_name}"'}
    return Response(out_bytes, mimetype="text/csv", headers=headers)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)

