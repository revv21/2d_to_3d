flowchart TD

%% =====================
%% INPUTS
%% =====================
A[Product Structure\nParts Subparts] 
B[Function Definitions]
C[Interfaces]
D[Operating Conditions]
E[Historical DFMEA]
F[Manufacturing Data\nOptional]
G[Test Coverage\nOptional]

%% =====================
%% SYSTEM MODEL
%% =====================
A --> H
B --> H
C --> H

H[System Graph Builder\nPart Function Interface] --> I[Knowledge Graph]

%% =====================
%% FUNCTION & FAILURE ANALYSIS
%% =====================
I --> J[Function Net Generator]

J --> K[Failure Mode Generator\nRules Templates AI]

D --> K
E --> K

K --> L[Effect Propagation Engine]

%% =====================
%% CAUSE & RISK SCORING
%% =====================
F --> M[Cause Inference Engine]
K --> M

L --> N[Severity Assignment\nStandards Based]
M --> O[Occurrence Estimation]
G --> P[Detection Estimation]

N --> Q
O --> Q
P --> Q

Q[Risk Scoring\nS O D to RPN]

%% =====================
%% DFMEA OUTPUT
%% =====================
Q --> R[DFMEA Table Generator]

R --> S[Engineer Review]

%% =====================
%% FTA GENERATION
%% =====================
S --> T[Approved DFMEA]

T --> U[Fault Tree Builder]
U --> V[Fault Tree Diagram]

%% =====================
%% LEARNING LOOP
%% =====================
S --> W[Feedback Store]
W --> K
W --> M