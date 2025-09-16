-- Initialize database for Cloud SQL Assistant
-- Create billing table with sample data

CREATE TABLE IF NOT EXISTS billing (
    invoice_month VARCHAR(7),
    account_id VARCHAR(50),
    subscription VARCHAR(100),
    service VARCHAR(100),
    resource_group VARCHAR(100),
    resource_id VARCHAR(200),
    region VARCHAR(50),
    usage_qty DECIMAL(10,2),
    unit_cost DECIMAL(10,4),
    cost DECIMAL(10,2)
);

-- Insert sample data for testing
INSERT INTO billing (invoice_month, account_id, subscription, service, resource_group, resource_id, region, usage_qty, unit_cost, cost) VALUES
('2022-12', 'acc-001', 'Production', 'Virtual Machines', 'rg-prod-web', 'vm-web-01', 'East US', 744.0, 0.096, 71.42),
('2022-12', 'acc-001', 'Production', 'Storage', 'rg-prod-data', 'storage-01', 'East US', 1024.0, 0.021, 21.50),
('2022-12', 'acc-001', 'Production', 'Virtual Network', 'rg-prod-network', 'vnet-01', 'East US', 1.0, 15.0, 15.0),
('2022-12', 'acc-001', 'Development', 'Virtual Machines', 'rg-dev-web', 'vm-dev-01', 'West US', 744.0, 0.048, 35.71),
('2022-12', 'acc-001', 'Development', 'Storage', 'rg-dev-data', 'storage-dev', 'West US', 512.0, 0.021, 10.75),
('2022-12', 'acc-002', 'Testing', 'Virtual Machines', 'rg-test', 'vm-test-01', 'Central US', 372.0, 0.024, 8.93),
('2022-12', 'acc-002', 'Testing', 'Backup', 'rg-test', 'backup-test', 'Central US', 100.0, 0.05, 5.0),
('2022-12', 'acc-002', 'Production', 'Load Balancer', 'rg-prod-lb', 'lb-prod-01', 'North Europe', 1.0, 25.0, 25.0),
('2022-12', 'acc-002', 'Production', 'Bandwidth', 'rg-prod-network', 'bandwidth-01', 'North Europe', 500.0, 0.087, 43.5),
('2022-12', 'acc-003', 'Production', 'Virtual Machines', 'rg-enterprise', 'vm-enterprise-01', 'East US', 744.0, 0.192, 142.85);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_billing_service ON billing(service);
CREATE INDEX IF NOT EXISTS idx_billing_region ON billing(region);
CREATE INDEX IF NOT EXISTS idx_billing_month ON billing(invoice_month);
CREATE INDEX IF NOT EXISTS idx_billing_cost ON billing(cost);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
