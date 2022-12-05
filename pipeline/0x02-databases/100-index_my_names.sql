-- Creates an index on the table `names`and the first letter of  `name`

-- DROP INDEX IF EXISTS names_first_letter_idx;

CREATE INDEX idx_name_first ON names (name(1));