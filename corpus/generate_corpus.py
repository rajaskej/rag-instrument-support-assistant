from pathlib import Path

from core.config import CORPUS_GEN_MODEL
from core.generation.llm_client import GeminiClient
from corpus.model_facts import MODELS

RAW_DIR = Path(__file__).parent / "raw"

MANUAL_TEMPLATE = """Write a technical instrument manual in Markdown for the {model_number}, a {tagline}.

Use exactly these top-level (#) sections in this order: Overview, Specifications, Calibration Procedure, Troubleshooting, Error Codes.

Specifications section: include a Markdown table with rows for the values below, using these exact values: {specs}.

Calibration Procedure section: a numbered list with exactly these steps, reworded naturally but preserving the order and meaning: {calibration_steps}.

Troubleshooting section: a Markdown table with two columns, Symptom and Error Code, that includes these exact symptom-to-code mappings: {symptoms}. You may add 1-2 additional plausible rows.

Error Codes section: one level-2 (##) subsection per code below, headed exactly with the code (e.g. "## E-104"), describing the cause and resolution using this information: {error_codes}.

Write in a plain technical documentation tone. Do not invent additional error codes beyond the ones listed. Output only the Markdown document, no commentary."""

SPEC_SHEET_TEMPLATE = """Write a one-page product specification sheet in Markdown for the {model_number}, a {tagline}.

Use a single top-level (#) heading "Specifications" followed by a Markdown table listing these exact values: {specs}.

Output only the Markdown document, no commentary."""

APP_REPORT_TEMPLATE = """Write a short application report in Markdown titled "{title}".

Use top-level (#) sections: Overview, Application, Method, Results.

The report should focus on: {focus}. It is for the {model_number} instrument. Keep it to about 500-700 words, plain technical tone, no fabricated numeric results tables — prose only.

Output only the Markdown document, no commentary."""


def _format_specs(model: dict) -> str:
    if model["family"] == "density_meter":
        return (
            f"density range {model['density_range']}, accuracy {model['accuracy']}, "
            f"temperature range {model['temp_range']}, sample volume {model['sample_volume']}"
        )
    return (
        f"torque range {model['torque_range']}, speed range {model['speed_range']}, "
        f"temperature range {model['temp_range']}"
    )


def _format_error_codes(model: dict) -> str:
    return "; ".join(f"{code}: {desc}" for code, desc in model["error_codes"].items())


def _format_symptoms(model: dict) -> str:
    return "; ".join(f'"{symptom}" -> {code}' for symptom, code in model["symptoms"])


def _format_steps(model: dict) -> str:
    return " | ".join(model["calibration_steps"])


def generate_doc(client, prompt: str) -> str:
    response = client.messages.create(
        model=CORPUS_GEN_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _write_if_missing(path: Path, client, prompt: str) -> None:
    if path.exists():
        print(f"  skip (already exists): {path.name}")
        return
    path.write_text(generate_doc(client, prompt))


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = GeminiClient()

    for model in MODELS:
        manual_prompt = MANUAL_TEMPLATE.format(
            model_number=model["model_number"],
            tagline=model["tagline"],
            specs=_format_specs(model),
            calibration_steps=_format_steps(model),
            symptoms=_format_symptoms(model),
            error_codes=_format_error_codes(model),
        )
        _write_if_missing(RAW_DIR / f"{model['model_number']}_manual.md", client, manual_prompt)

        spec_prompt = SPEC_SHEET_TEMPLATE.format(
            model_number=model["model_number"], tagline=model["tagline"], specs=_format_specs(model)
        )
        _write_if_missing(RAW_DIR / f"{model['model_number']}_spec_sheet.md", client, spec_prompt)

        if "app_report" in model:
            report_prompt = APP_REPORT_TEMPLATE.format(
                title=model["app_report"]["title"],
                focus=model["app_report"]["focus"],
                model_number=model["model_number"],
            )
            _write_if_missing(RAW_DIR / f"{model['model_number']}_app_report.md", client, report_prompt)

        print(f"Generated docs for {model['model_number']}")


if __name__ == "__main__":
    main()
