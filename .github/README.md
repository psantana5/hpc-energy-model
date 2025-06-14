# CI/CD Pipeline Documentation

This directory contains GitHub Actions workflows for the HPC Energy Model project. The CI/CD pipeline provides comprehensive testing, validation, security scanning, and automated deployment capabilities.

## 🔄 Workflows Overview

### 1. CI/CD Pipeline (`ci.yml`)
**Triggers:** Push/PR to main/develop branches  
**Purpose:** Main continuous integration pipeline

**Jobs:**
- **Lint and Validate**: Code quality checks (flake8, black, isort, bandit, safety)
- **Test Python Components**: Unit tests and import validation
- **Validate Scripts**: Shell script syntax and AWS configuration validation
- **Security Scan**: Trivy vulnerability scanning and secret detection
- **Infrastructure Validation**: Prometheus, Grafana, Fluentd, Nginx config validation
- **Documentation**: Markdown linting and required file checks

### 2. Deployment Pipeline (`deploy.yml`)
**Triggers:** Push to main, tags, manual dispatch  
**Purpose:** Automated deployment to staging and production

**Jobs:**
- **Validate Deployment**: Pre-deployment configuration validation
- **Build and Test**: Comprehensive testing before deployment
- **Deploy Staging**: Automated staging environment deployment
- **Deploy Production**: Production deployment (tags/manual only)
- **Cleanup on Failure**: Automatic resource cleanup on failed deployments

**Environments:**
- `staging`: Deploys on main branch pushes
- `production`: Deploys on version tags or manual trigger

### 3. Monitoring and Infrastructure Tests (`monitoring-test.yml`)
**Triggers:** Changes to infrastructure/monitoring files, weekly schedule  
**Purpose:** Specialized testing for monitoring stack

**Jobs:**
- **Test Monitoring Stack**: Docker Compose, Prometheus, Grafana validation
- **Test SLURM Integration**: SLURM configuration and script validation
- **Integration Test**: End-to-end monitoring stack testing
- **Security Infrastructure Scan**: Infrastructure-specific security checks

### 4. Dependency Management (`dependency-update.yml`)
**Triggers:** Weekly schedule, manual dispatch  
**Purpose:** Automated dependency updates and security monitoring

**Jobs:**
- **Check Dependencies**: Security auditing with pip-audit and safety
- **Update Dependencies**: Automated dependency updates (security/minor/all)
- **Docker Security Scan**: Container vulnerability scanning

**Update Types:**
- `security`: Only update packages with known vulnerabilities
- `minor`: Update to latest minor versions
- `all`: Update all packages to latest versions

### 5. Performance Testing (`performance-test.yml`)
**Triggers:** Changes to workloads/analysis/api, weekly schedule, manual dispatch  
**Purpose:** Performance benchmarking and regression testing

**Jobs:**
- **CPU Benchmark**: CPU-intensive workload performance testing
- **I/O Benchmark**: I/O-intensive workload performance testing
- **Mixed Benchmark**: Mixed workload performance testing
- **API Performance Test**: API load testing and response time validation
- **Performance Summary**: Consolidated performance reporting

### 6. Release and Changelog (`release.yml`)
**Triggers:** Version tags, manual dispatch  
**Purpose:** Automated release creation and changelog generation

**Jobs:**
- **Validate Release**: Version format validation and tag checking
- **Run Tests**: Full test suite execution
- **Build Artifacts**: Source distribution and deployment package creation
- **Generate Changelog**: Automated changelog generation from git history
- **Create Release**: GitHub release creation with artifacts
- **Notify Deployment**: Release notification and deployment readiness

## 🔧 Configuration Files

### `.markdownlint.json`
Markdown linting configuration for documentation quality:
- Line length: 120 characters
- Allows HTML elements: `<details>`, `<summary>`, `<br>`
- Relaxed heading rules for better documentation flexibility

## 🚀 Usage Guide

### Running CI/CD Locally

1. **Install dependencies:**
   ```bash
   pip install flake8 black isort bandit safety yamllint shellcheck-py
   ```

2. **Run linting:**
   ```bash
   flake8 .
   black --check .
   isort --check-only .
   ```

3. **Security checks:**
   ```bash
   bandit -r .
   safety check
   ```

4. **Validate configurations:**
   ```bash
   yamllint *.yml *.yaml
   find . -name "*.sh" -exec shellcheck {} \;
   ```

### Manual Deployment

1. **Staging deployment:**
   ```bash
   # Trigger via GitHub Actions UI
   # Or push to main branch
   ```

2. **Production deployment:**
   ```bash
   # Create and push a version tag
   git tag v1.0.0
   git push origin v1.0.0
   ```

3. **Manual deployment trigger:**
   - Go to Actions tab in GitHub
   - Select "Deployment Pipeline"
   - Click "Run workflow"
   - Choose environment and cluster name

### Performance Testing

1. **Run specific benchmark:**
   ```bash
   # Via GitHub Actions UI
   # Select benchmark type: cpu, io, mixed, api, or all
   ```

2. **Local performance testing:**
   ```bash
   cd workloads/cpu-intensive
   python cpu_benchmark.py --duration 60
   
   cd ../io-intensive
   python io_benchmark.py --duration 60
   
   cd ../mixed
   python mixed_benchmark.py --duration 60
   ```

### Dependency Updates

1. **Security updates only:**
   - Automatically runs weekly
   - Manual trigger: Actions → "Dependency Management" → "security"

2. **Minor updates:**
   - Manual trigger with "minor" option
   - Updates to latest minor versions

3. **All updates:**
   - Manual trigger with "all" option
   - Updates all packages to latest versions

## 🔒 Security Features

### Automated Security Scanning
- **Code Security**: Bandit for Python security issues
- **Dependency Security**: Safety and pip-audit for known vulnerabilities
- **Container Security**: Trivy for Docker image vulnerabilities
- **Secret Detection**: TruffleHog for exposed secrets
- **Infrastructure Security**: Configuration validation and best practices

### Security Thresholds
- High/Critical vulnerabilities fail the build
- Security updates are automatically prioritized
- Failed deployments trigger automatic cleanup

## 📊 Monitoring and Reporting

### Artifacts and Reports
- **Security Reports**: JSON format, 30-day retention
- **Performance Reports**: Markdown format, 90-day retention
- **Build Artifacts**: Release packages and distributions
- **Test Results**: Coverage reports and test outputs

### Notifications
- **Failed Builds**: Automatic GitHub issue creation (configurable)
- **Security Alerts**: Weekly dependency security reports
- **Performance Regression**: Performance threshold alerts
- **Release Notifications**: Automated release announcements

## 🛠️ Customization

### Environment Variables
Configure in GitHub repository settings → Secrets and variables:

**Required for AWS Deployment:**
- `AWS_ACCESS_KEY_ID`: AWS access key for staging
- `AWS_SECRET_ACCESS_KEY`: AWS secret key for staging
- `AWS_ACCESS_KEY_ID_PROD`: AWS access key for production
- `AWS_SECRET_ACCESS_KEY_PROD`: AWS secret key for production

**Optional:**
- `SLACK_WEBHOOK_URL`: For Slack notifications
- `CODECOV_TOKEN`: For code coverage reporting

### Workflow Customization

1. **Modify trigger conditions:**
   ```yaml
   on:
     push:
       branches: [ main, develop, feature/* ]
       paths:
         - 'src/**'
         - 'tests/**'
   ```

2. **Adjust performance thresholds:**
   ```yaml
   - name: Check performance thresholds
     run: |
       if [ $RESPONSE_TIME -gt 2000 ]; then
         echo "Performance regression detected"
         exit 1
       fi
   ```

3. **Add custom validation:**
   ```yaml
   - name: Custom validation
     run: |
       # Add your custom validation logic
       ./scripts/custom-validation.sh
   ```

## 🐛 Troubleshooting

### Common Issues

1. **AWS Deployment Failures:**
   - Check AWS credentials in repository secrets
   - Verify IAM permissions for ParallelCluster
   - Check AWS service limits and quotas

2. **Performance Test Failures:**
   - Review performance thresholds
   - Check system resource availability
   - Verify benchmark script compatibility

3. **Security Scan Failures:**
   - Review security reports in artifacts
   - Update vulnerable dependencies
   - Check for exposed secrets in code

4. **Docker Issues:**
   - Verify Docker Compose syntax
   - Check image availability and versions
   - Review container resource requirements

### Debug Mode
Enable debug logging by adding to workflow:
```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

## 📚 Best Practices

1. **Branch Protection:**
   - Require PR reviews
   - Require status checks to pass
   - Restrict force pushes

2. **Secret Management:**
   - Use GitHub Secrets for sensitive data
   - Rotate credentials regularly
   - Use environment-specific secrets

3. **Testing Strategy:**
   - Run fast tests on every commit
   - Run comprehensive tests on main branch
   - Run performance tests on schedule

4. **Deployment Strategy:**
   - Always deploy to staging first
   - Use blue-green deployments for production
   - Implement automatic rollback on failure

5. **Monitoring:**
   - Monitor workflow execution times
   - Track success/failure rates
   - Set up alerts for critical failures

## 🔗 Related Documentation

- [AWS Deployment Guide](../AWS_DEPLOYMENT.md)
- [General Deployment Guide](../DEPLOYMENT.md)
- [Enhanced Features](../ENHANCED_FEATURES.md)
- [Project README](../README.md)

---

*This CI/CD pipeline is designed to ensure code quality, security, and reliable deployments for the HPC Energy Model project. For questions or improvements, please open an issue or submit a pull request.*