SET SQL_SAFE_UPDATES = 0;
DELETE FROM stock_metrics WHERE tag in ('revenue_growth_1yr', 'revenue_growth_3yr', 'operating_profit_growth_3yr',
'operating_profit_growth_1yr', 'mc',
       'price_to_equity', 'profit_margin', 'interest_coverage_ratio',
       'equity_per_share', 'roe', 'roa', 'debt_to_equity',
       'current_ratio', 'debt_to_asset', 'perc_dividend','gross_npa_perc','net_npa_perc');
SET SQL_SAFE_UPDATES = 1;