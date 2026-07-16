.PHONY: report-assets dashboard dm-test

report-assets:
	uv run python scripts/export_report_assets.py

dashboard:
	uv run --extra dashboard streamlit run src/energy_price_forecast/dashboard/app.py

dm-test:
	uv run python scripts/run_dm_test.py
