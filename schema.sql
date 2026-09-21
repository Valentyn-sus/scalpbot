CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 3.1 Master Task Records
CREATE TABLE IF NOT EXISTS crawler_tasks (
    task_id UUID PRIMARY KEY,
    domain VARCHAR(255) NOT NULL,
    source VARCHAR(50),
    status VARCHAR(50) DEFAULT 'PENDING',
    final_result JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3.2 Audit Log History
CREATE TABLE IF NOT EXISTS crawler_task_history (
    id BIGSERIAL PRIMARY KEY,
    task_id UUID REFERENCES crawler_tasks(task_id) ON DELETE CASCADE,
    step_name VARCHAR(100) NOT NULL,
    step_status VARCHAR(50) NOT NULL,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_crawler_history_task_id ON crawler_task_history(task_id);

-- 3.3 Bot Credentials
CREATE TABLE IF NOT EXISTS bot_credentials (
    domain_cluster VARCHAR(255) PRIMARY KEY,
    auth_method VARCHAR(20) DEFAULT 'email',
    email VARCHAR(255),
    password VARCHAR(255),
    session_cookies JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);