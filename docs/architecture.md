# System Architecture

## Medallion Architecture

```text
Cybersecurity Dataset
        |
        v
     BRONZE
   Raw Data
        |
        v
     SILVER
Cleaned & Validated Data
        |
        v
      GOLD
ML & Analytics Data
        |
        v
   ML Detection
        |
        v
  Scalability Testing
        |
        v
   Benchmarking