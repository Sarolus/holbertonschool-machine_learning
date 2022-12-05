-- Displays the average temperature

SELECT city, AVG (value) as average FROM temperatures GROUP BY city ORDER BY city, average;