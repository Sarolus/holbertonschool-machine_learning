-- Creates an index on the table `names`and the first letter of  `name`

-- DROP INDEX IF EXISTS names_first_letter_idx;

CREATE INDEX names_first_letter_idx ON names (name(1));