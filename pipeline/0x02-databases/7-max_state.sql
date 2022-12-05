-- Displays the max temperature for each state in the third_table

SELECT state, MAX(value) as max FROM temperatures GROUP BY state ORDER BY state, max;