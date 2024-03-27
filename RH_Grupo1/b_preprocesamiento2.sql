----Los comentarios en sql se ponen con --
---después de finalizar cada consulta se pone ;

------ se filtran empleados que no tengan evaluación de 2023 porque se retiraron antes de la evaluación y los que no tenían antes de 2023 porque entraron ese año

drop table if exists performance2;

-- Con la cláusula DISTINCT aseguramos que solamente hayan combinaciones unicas
create table former_employees_2016 as
select DISTINCT
EmployeeID,
retirementDate,
retirementType,
resignationReason
from all_employees
where strftime('%Y', retirementDate) = '2016';


