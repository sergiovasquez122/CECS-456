select * from ProductLines

select * from Products

select * from Employees

select * from Offices

select * from Customers

select * from Orders

select * from OrderDetails

select * from Payments

select customerName from Customers order by customerName asc

select distinct status from Orders

select lastName, firstName from Employees order by lastName, firstName

select distinct jobTitle from Employees

select productScale, productName from Products

select distinct territory from Offices

select contactLastName, contactFirstName, creditLimit from Customers where creditLimit > 50000

select * from Customers where creditLimit = 0

select * from Offices where not Country in ('USA')

select * from Orders where orderDate between '06/16/2014' and '07/07/2014'

select * from Products where quantityInStock < 1000

select * from Orders where shippedDate > requiredDate

select * from Customers where customerName like '%Mini%'

select * from Products where productVendor = 'Highway 66 Mini Classics'

select * from Products where not productVendor = 'Highway 66 Mini Classics'

select * from Employees where reportsTo is null

select * from OrderDetails natural join Orders where OrderNumber in (10270, 10272, 10279)

select distinct productLine, productVendor from ProductLines natural join Products

select * from Customers Inner Join Offices on
Customers."STATE" = Offices."STATE"

select * from Customers inner join Offices 
on Customers."STATE" = Offices."STATE"


