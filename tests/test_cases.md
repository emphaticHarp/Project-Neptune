# Project Neptune: Test Cases

## Hardware Testing

### River Station Tests

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| RS-001 | Power system functionality | System powers on and maintains charge | Battery voltage > 7V after 24h operation |
| RS-002 | Water level sensor accuracy | Readings match manual measurements | Error < 1cm across 10-200cm range |
| RS-003 | Water flow sensor calibration | Flow rate matches control flow | Error < 5% across 1-30L/min range |
| RS-004 | Environmental sensor readings | Temperature and humidity values | Within ±1°C and ±3% of reference device |
| RS-005 | LoRa communication range | Successful data transmission | Packet loss < 5% at 5km distance |
| RS-006 | GSM fallback activation | Switch to GSM when LoRa fails | Automatic failover within 2 minutes |
| RS-007 | Weatherproofing | No water ingress after exposure | IP67 rating verified after rain test |
| RS-008 | Data sampling frequency | Regular data collection | 1 sample every 5 minutes ±10 seconds |
| RS-009 | Low power mode | Reduced power consumption | Current draw < 50mA in sleep mode |

### Main Station Tests

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| MS-001 | Server uptime | Continuous operation | 99.9% uptime over 7-day period |
| MS-002 | Data reception | All station data received | No data gaps > 15 minutes |
| MS-003 | Database storage | Data correctly stored | Query results match input data |
| MS-004 | Alert generation | Alerts for threshold violations | Alert within 2 minutes of condition |
| MS-005 | UPS functionality | Continued operation during power outage | 8+ hours operation on battery |
| MS-006 | Multi-station handling | Process data from all stations | Support for 10+ simultaneous stations |
| MS-007 | System monitoring | Self-diagnostics and reporting | Accurate reporting of system status |

## Software Testing

### Backend API Tests

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| API-001 | Station data endpoint | Return latest station data | 200 OK with valid JSON response |
| API-002 | Historical data query | Return data for specified period | Complete dataset with correct timestamps |
| API-003 | Alert history endpoint | Return recent alerts | Correctly formatted alert objects |
| API-004 | System status endpoint | Return component status | Accurate status for all components |
| API-005 | Authentication | Reject unauthorized requests | 401 Unauthorized for invalid credentials |
| API-006 | Rate limiting | Prevent API abuse | 429 Too Many Requests after threshold |
| API-007 | Error handling | Graceful handling of bad requests | Appropriate error codes and messages |

### Dashboard Tests

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| UI-001 | Responsive design | Proper display on various devices | Correct layout on desktop, tablet, mobile |
| UI-002 | Real-time updates | Dashboard updates without refresh | New data appears within 10 seconds |
| UI-003 | Map functionality | Interactive station markers | Correct location and popup information |
| UI-004 | Gauge visualization | Accurate representation of values | Gauge position matches data value |
| UI-005 | Alert display | Prominent alert notifications | Visible alert within 5 seconds of generation |
| UI-006 | Historical data charts | Data visualization for trends | Chart matches database values |
| UI-007 | Cross-browser compatibility | Works in major browsers | Identical functionality in Chrome, Firefox, Safari |
| UI-008 | Accessibility | Meets WCAG 2.1 AA standards | Passes automated accessibility tests |

## AI Model Testing

### Flood Detection Tests

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| FD-001 | Detection accuracy | Identify flooding in images | >85% accuracy on test dataset |
| FD-002 | False positive rate | Minimize incorrect detections | <10% false positive rate |
| FD-003 | Classification accuracy | Correct flood severity level | >80% correct classification |
| FD-004 | Processing speed | Real-time analysis | <100ms per frame on target hardware |
| FD-005 | Low light performance | Function in poor lighting | >70% accuracy in low light conditions |
| FD-006 | Weather interference | Handle rain, fog, snow | Maintain >75% accuracy in adverse weather |

### Person Detection Tests

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| PD-001 | Detection accuracy | Identify people in flood scenes | >80% detection rate on test dataset |
| PD-002 | False positive rate | Minimize incorrect detections | <5% false positive rate |
| PD-003 | Partial occlusion handling | Detect partially visible people | >70% detection with 50% occlusion |
| PD-004 | Distance performance | Detect people at various distances | Effective range up to 50m |
| PD-005 | Processing speed | Real-time analysis | <150ms per frame on target hardware |
| PD-006 | Multiple person detection | Identify groups of people | Correctly count up to 10 people per frame |

## Integration Testing

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| INT-001 | End-to-end data flow | Data from sensor to dashboard | Complete pipeline with <30s latency |
| INT-002 | Alert notification system | SMS and email alerts | Notifications sent within 3 minutes |
| INT-003 | System recovery | Recovery after power/network outage | Automatic recovery with no data loss |
| INT-004 | AI integration | Detection results in dashboard | Detection results displayed within 2 minutes |
| INT-005 | Multi-component stress test | System under maximum load | Stable operation at 200% normal load |
| INT-006 | Long-term stability | Extended operation period | Stable for 30+ days continuous operation |

## Performance Testing

| Test ID | Description | Expected Result | Pass Criteria |
|---------|-------------|-----------------|---------------|
| PERF-001 | Database query performance | Fast data retrieval | <500ms for standard queries |
| PERF-002 | Dashboard loading time | Quick initial load | <3s first load, <1s subsequent loads |
| PERF-003 | API response time | Fast API responses | <200ms average response time |
| PERF-004 | Concurrent user handling | Support multiple users | 50+ simultaneous users without degradation |
| PERF-005 | Data processing throughput | Handle high data volume | Process 1000+ readings per minute |
| PERF-006 | Memory usage | Efficient resource utilization | <70% memory usage under normal load |
| PERF-007 | CPU utilization | Efficient processing | <50% CPU usage under normal load |
