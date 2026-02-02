#!/bin/bash
set -e

echo "🚀 Starting Portfolio Tracker..."

# Wait for PostgreSQL to be ready (healthcheck should handle this, but double-check)
echo "⏳ Waiting for database to be ready..."
for i in {1..30}; do
    if python -c "from sqlalchemy import create_engine; import os; engine = create_engine(os.environ['PORTFOLIO_DB_URL']); engine.connect()" 2>/dev/null; then
        echo "✅ Database is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Database failed to become ready in time"
        exit 1
    fi
    echo "   Attempt $i/30 - waiting..."
    sleep 2
done

# Initialize database tables if not already done
echo "🔧 Initializing database schema..."
python -c "
from db.session import init_db
db = init_db()
print('✅ Database schema initialized')
" || {
    echo "❌ Database initialization failed"
    exit 1
}

echo "🎯 Starting Streamlit dashboard..."
exec streamlit run ui/app.py --server.address=0.0.0.0 --server.port=8501
