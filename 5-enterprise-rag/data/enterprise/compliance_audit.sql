CREATE TABLE compliance_controls (
  control_id TEXT,
  framework TEXT,
  owner TEXT,
  status TEXT,
  finding TEXT
);

INSERT INTO compliance_controls VALUES
('SOC2-CC6.1', 'SOC 2', 'security', 'partial', 'Privileged access review evidence was missing for two contractors');

INSERT INTO compliance_controls VALUES
('ISO27001-A.8.12', 'ISO 27001', 'platform', 'pass', 'Data leakage prevention policy is enabled for managed endpoints');

INSERT INTO compliance_controls VALUES
('HIPAA-164.312', 'HIPAA Security Rule', 'compliance', 'partial', 'Audit log retention meets policy but alert review sign-off was delayed');
