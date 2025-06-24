"""update loandata model

Revision ID: 41fe02c9e254
Revises: adc807458e2b
Create Date: 2025-06-23 16:02:32.999602

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '41fe02c9e254'
down_revision: Union[str, None] = 'adc807458e2b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema (manual for SQLite)."""
    # 1. Créer une nouvelle table temporaire avec la bonne structure
    op.create_table(
        'loandatas_new',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('age', sa.Integer),
        sa.Column('taille', sa.Float),
        sa.Column('poids', sa.Float),
        sa.Column('nb_enfants', sa.Integer),
        sa.Column('quotient_caf', sa.Float),
        sa.Column('sport_licence', sa.String),
        sa.Column('niveau_etude', sa.Integer),  # nouveau type
        sa.Column('region', sa.String),
        sa.Column('smoker', sa.String),
        sa.Column('revenu_estime_mois', sa.Integer),
        sa.Column('situation_familiale', sa.String),
        sa.Column('risque_personnel', sa.Float),
        sa.Column('loyer_mensuel', sa.Float),
        sa.Column('montant_pret', sa.Float),
    )
    # 2. Copier les données (attention: conversion possible à faire si besoin)
    op.execute("""
        INSERT INTO loandatas_new (id, age, taille, poids, nb_enfants, quotient_caf, sport_licence, niveau_etude, region, smoker, revenu_estime_mois, situation_familiale, risque_personnel, loyer_mensuel, montant_pret)
        SELECT id, age, taille, poids, nb_enfants, quotient_caf, sport_licence, niveau_etude, region, smoker, revenu_estime_mois, situation_familiale, risque_personnel, loyer_mensuel, montant_pret FROM loandatas
    """)
    # 3. Supprimer l’ancienne table
    op.drop_table('loandatas')
    # 4. Renommer la nouvelle table
    op.rename_table('loandatas_new', 'loandatas')


def downgrade() -> None:
    # Downgrade non implémenté (à faire si besoin)
    pass
