-- Add new columns to rules table
ALTER TABLE rules ADD COLUMN IF NOT EXISTS is_draft BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE rules ADD COLUMN IF NOT EXISTS confidence INTEGER NULL; 