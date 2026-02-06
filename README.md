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
%% AUTOMATION
%% =====================
H[System Graph Builder]
I[Knowledge Graph]
J[Function Net Generator]
K[Failure Mode Generator]
L[Effect Propagation Engine]
M[Cause Inference Engine]
N[Severity Assignment]
O[Occurrence Estimation]
P[Detection Estimation]
Q[Risk Scoring]
R[DFMEA Table Generator]
U[Fault Tree Builder]

%% =====================
%% HUMAN INTERVENTION
%% =====================
S[Engineer Review\nValidation Approval]

%% =====================
%% OUTPUTS
%% =====================
T[Approved DFMEA]
V[Fault Tree Diagram]

%% =====================
%% FLOW
%% =====================
A --> H
B --> H
C --> H

H --> I
I --> J
J --> K

D --> K
E --> K

K --> L
K --> M

L --> N
M --> O
G --> P

N --> Q
O --> Q
P --> Q

Q --> R
R --> S

S --> T
T --> U
U --> V

%% =====================
%% STYLES
%% =====================
classDef input fill:#cce5ff,stroke:#004085,stroke-width:1px
classDef auto fill:#d4edda,stroke:#155724,stroke-width:1px
classDef human fill:#fff3cd,stroke:#856404,stroke-width:1px
classDef output fill:#e2d6f3,stroke:#4b2c83,stroke-width:1px

%% =====================
%% CLASS ASSIGNMENTS
%% =====================
class A,B,C,D,E,F,G input
class H,I,J,K,L,M,N,O,P,Q,R,U auto
class S human
class T,V output