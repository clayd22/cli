# Data Platform Context

## Overview
Data platform contains multiple schemas for different stages of ETL process and business domains, including raw, staging, marts and main schemas.
## Key Tables
- marts.fct_orders: Contains detailed order information including transaction date, customer details, product details, pricing, and order status.
- raw.transactions: Another table containing transaction details, likely similar to marts.fct_orders but in a rawer format.
- marts.dim_customers and marts.dim_products: Contain customer and product details respectively.
## Relationships
- marts.fct_orders is linked to marts.dim_customers and marts.dim_products via user_id and product_id, respectively.
- raw.transactions can also be linked to users and products tables in a similar manner.
- Campaign-related data can be linked across campaigns and orders tables using campaign_id.
## Common Patterns
<!-- Useful query patterns discovered -->

## Notes
Use 'marts' schema for up-to-date queries, rather than 'staging_marts' which may contain stale data.