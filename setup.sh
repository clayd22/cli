#!/bin/bash
# setup.sh - One-command setup for the mini data platform

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIRFLOW_DIR="$PROJECT_ROOT/airflow"
AIRFLOW_DB_PATH="$AIRFLOW_DIR/airflow.db"

echo "🚀 Setting up Mini Data Platform..."
echo ""

# Step 0: Configure Airflow with correct absolute path
echo "⚙️  Configuring Airflow..."
# Use sed to replace the connection string with absolute path
sed -i.bak "s|sql_alchemy_conn = .*|sql_alchemy_conn = sqlite:///$AIRFLOW_DB_PATH|g" "$AIRFLOW_DIR/airflow.cfg"
rm -f "$AIRFLOW_DIR/airflow.cfg.bak"
echo "✓ Airflow configured with database at: $AIRFLOW_DB_PATH"
echo ""

# Step 1: Generate synthetic data
echo "📊 Step 1/3: Generating synthetic data..."
cd "$PROJECT_ROOT"
uv run python scripts/generate_all.py
echo "✓ Data generated"
echo ""

# Step 2: Initialize Airflow
echo "⚙️  Step 2/3: Initializing Airflow metadata database..."
cd "$AIRFLOW_DIR"
export AIRFLOW_HOME="$AIRFLOW_DIR"
uv run airflow db migrate 2>&1 | grep -E "(Performing upgrade|Database migrating done|ERROR)" || true
if [ $? -eq 0 ]; then
    echo "✓ Airflow initialized"
else
    echo "⚠️  Airflow initialization encountered an issue"
fi
echo ""

# Step 3: Run ingestion DAGs
echo "📥 Step 3/3: Running ingestion and transformation..."
export AIRFLOW_HOME="$AIRFLOW_DIR"
uv run python dags/ingest_products.py
uv run python dags/ingest_users.py
uv run python dags/ingest_transactions.py
uv run python dags/ingest_campaigns.py
uv run python dags/ingest_pageviews.py

# Run dbt transformations
uv run python dags/run_dbt.py

echo ""
echo "✅ Setup complete! Your data platform is ready."
echo ""
echo "Next steps:"
echo "  • View dashboards:"
echo "      cd evidence"
echo "      npm install           # First time only"
echo "      npm run sources       # Build data sources"
echo "      npm run dev           # Start dev server"
echo ""
echo "  • Query warehouse: duckdb warehouse/data.duckdb"
echo ""

