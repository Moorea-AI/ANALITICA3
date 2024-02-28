-- Se solicita al equipo de analítica proponer estrategias que permitan tomar acciones para 
-- reducir el porcentaje de retiros utilizando la información disponible que se tiene de los empleados. 
-- Asuma que la fecha actual es 1 de enero de 2017, y el plan y estrategia que proponga, son para reducir
--  los retiros de los empleados durante el año 2017, para el cuál se espera que el porcentaje de retiros 
--  no super el 12% de los empleados.--

-- Como estamos parados en 2017, 1 de Enero. Requerimos saber cuántos empleados se retiraron en 2016, 
-- para esto seleccionamos el ID del empleado, fechas, tipo de retiro y porque se retiró


drop table if exists retirement_2016;

create table retirement_2016 as
select
EmployeeID,
retirementDate,
retirementType,
resignationReason
from all_employees
where strftime('%Y', retirementDate) = '2016';