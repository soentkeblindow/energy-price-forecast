.PHONY: report-assets dashboard

report-assets:
	uv run python scripts/export_report_assets.py

dashboard:
	uv run --extra dashboard streamlit run src/energy_price_forecast/dashboard/app.py
