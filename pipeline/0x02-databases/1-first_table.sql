-- Creates the first table into the database db_0 if it doesn't exist

CREATE TABLE IF NOT EXISTS first_table (
  id INT,
  name VARCHAR(255) NOT NULL
);