SET SQL_SAFE_UPDATES = 0;
DELETE FROM stock_metrics WHERE tag in ('ebitda', 'debt', 'ev_to_ebitda', 'en_value', 'mc',
       'price_to_equity', 'profit_margin', 'interest_coverage_ratio',
       'equity_per_share', 'roe', 'roa', 'debt_to_equity',
       'current_ratio', 'debt_to_asset', 'perc_dividend','gross_npa_perc','net_npa_perc');
SET SQL_SAFE_UPDATES = 1;