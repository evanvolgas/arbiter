"""Create arbiter schema and tables

Revision ID: 94176ccd39d2
Revises: 
Create Date: 2025-11-23 18:38:57.245854

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '94176ccd39d2'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create arbiter schema and evaluation results tables."""
    # Create arbiter schema
    op.execute("CREATE SCHEMA IF NOT EXISTS arbiter")

    # Create evaluation_results table
    op.execute("""
        CREATE TABLE arbiter.evaluation_results (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            result_data JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            overall_score FLOAT,
            evaluators TEXT[],
            model TEXT
        )
    """)

    # Create batch_evaluation_results table
    op.execute("""
        CREATE TABLE arbiter.batch_evaluation_results (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            result_data JSONB NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            total_items INT,
            successful_items INT,
            failed_items INT,
            avg_score FLOAT
        )
    """)

    # Create indexes for common queries
    op.execute("""
        CREATE INDEX idx_eval_results_created
        ON arbiter.evaluation_results (created_at DESC)
    """)
    op.execute("""
        CREATE INDEX idx_eval_results_score
        ON arbiter.evaluation_results (overall_score)
    """)
    op.execute("""
        CREATE INDEX idx_batch_results_created
        ON arbiter.batch_evaluation_results (created_at DESC)
    """)


def downgrade() -> None:
    """Drop arbiter schema and all tables."""
    op.execute("DROP SCHEMA IF EXISTS arbiter CASCADE")
