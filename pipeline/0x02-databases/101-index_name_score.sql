-- Creates an index on the table `names` and the first letter of `name` and the score

CREATE INDEX names_first_letter_score_idx ON names (name(1), score);