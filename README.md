# Safiri ETA Prediction

## Problem

Freight shipments move through five sequential stages — origin departure, port
arrival, customs clearance, inland transport, and final delivery — and a delay
incurred at an early stage propagates downstream, compressing the time
available for every stage that follows. This prototype predicts a shipment's
final ETA and its delay risk partway through that journey, while there is still
time to act on the forecast. Beyond the prediction itself, it explains *why*:
which operational factors drive the estimate, and how much of it is attributable
to delay already accumulated at upstream stages. The goal is a forecast an
operations team can interrogate and trust, not just a number.

## Assumptions

*Placeholders — each is resolved and expanded in a later stage of the build.*

- Prediction cutoff definition: TBD
- Delay threshold definition: TBD
- Congestion/weather/customs complexity scales: TBD
- Missing data handling approach: TBD
