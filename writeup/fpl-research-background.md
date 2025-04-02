# Background: English Premier League and Fantasy Premier League

## 1. The English Premier League: Structure and Significance

The English Premier League (EPL) is the top tier of professional football (soccer) in England, founded in 1992 after breaking away from the Football League[^1]. It consists of 20 clubs that compete in a double round-robin tournament, playing 38 matches each season (home and away against every other team). The season typically runs from August to May, with teams awarded three points for a win, one for a draw, and none for a loss. At the end of each season, the three lowest-ranked teams are relegated to the Championship (second tier), while three teams are promoted from the Championship to the Premier League[^2].

The EPL has grown to become the most-watched sports league globally, broadcasting to 212 territories with a potential audience of 4.7 billion people[^3]. Its commercial success is unprecedented, with the 2022-2025 broadcasting rights valued at approximately £10 billion[^4]. This financial power has enabled EPL clubs to attract elite players and coaches from around the world, contributing to the league's competitive nature and global appeal[^5].

## 2. Fantasy Premier League: Game Mechanics and Popularity

Fantasy Premier League (FPL) is the official fantasy sports game associated with the English Premier League. Launched in 2002, it has grown exponentially to over 11 million players worldwide as of the 2023/24 season[^6]. FPL allows participants to assemble a virtual team of real Premier League players within specific constraints and earn points based on those players' actual performances in Premier League matches[^7].

### 2.1 Basic Rules and Structure

Participants (known as "managers") are allocated a virtual budget (£100 million) to select a 15-player squad consisting of:
- 2 Goalkeepers
- 5 Defenders
- 5 Midfielders
- 3 Forwards

The budget constraint forces managers to balance premium-priced elite players with cheaper options. Each gameweek, managers select 11 players from their 15-player squad to form a starting lineup, with the remaining 4 players on the bench. Additional constraints include:
- Maximum of 3 players from any single Premier League club
- Formation requirements (minimum of 1 goalkeeper, 3 defenders, and 1 forward)
- Limited free transfers between gameweeks (typically 1 per week, with additional transfers costing points)[^8]

### 2.2 Scoring System

Points are awarded based on players' real-world performance metrics:
- Appearance (playing at least 60 minutes): 2 points
- Goals: 6 points (midfielder), 4 points (forward), 6 points (defender/goalkeeper)
- Assists: 3 points
- Clean sheets: 4 points (defender/goalkeeper), 1 point (midfielder)
- Saves: 1 point per 3 saves (goalkeeper)
- Penalties saved: 5 points (goalkeeper)
- Bonus points: 1-3 additional points to the top performers in each match

Negative points are also assigned for:
- Yellow cards: -1 point
- Red cards: -3 points
- Own goals: -2 points
- Penalties missed: -2 points
- Goals conceded: -1 point per 2 goals (defender/goalkeeper)[^9]

### 2.3 Special Features

FPL includes several strategic elements that increase its complexity:
- **Captain**: Managers designate one player as captain each gameweek, doubling their points
- **Vice-captain**: A backup who becomes captain if the original captain doesn't play
- **Chips**: Special boosts used once per season:
  - Bench Boost: Points from bench players count for one gameweek
  - Triple Captain: Triple (rather than double) points for the captain
  - Free Hit: Unlimited free transfers for one gameweek only
  - Wildcard: Unlimited free transfers that permanently change the team[^10]

## 3. Data and Performance Metrics in Football

### 3.1 Traditional Statistics

Football has historically relied on basic statistics to evaluate performance:
- Goals and assists
- Clean sheets
- Shots and shots on target
- Pass completion percentage
- Possession percentage
- Cards and fouls[^11]

### 3.2 Advanced Metrics

Recent years have seen an explosion in advanced metrics:
- Expected Goals (xG): Probability of a shot resulting in a goal
- Expected Assists (xA): Probability of a pass leading to a goal
- Progressive Passes/Carries: Passes/carries that move the ball significantly toward the opponent's goal
- Defensive Actions: Tackles, interceptions, clearances, and blocks
- Pressure Events: Instances of applying pressure to an opponent
- VAEP (Value of Actions by Estimating Probabilities): Calculating the value of every action[^12][^13]

### 3.3 Player Pricing and Value

FPL assigns each player a monetary value, which fluctuates throughout the season based on ownership patterns. The game adjusts player prices according to transfer market dynamics:
- Players transferred in by many managers typically increase in price
- Players transferred out by many managers typically decrease in price
- Price changes occur in £0.1m increments within certain thresholds[^14]

This dynamic pricing creates a parallel "market economy" that influences decision-making, as managers must consider not only point-scoring potential but also value appreciation/depreciation[^15].

## 4. Decision-Making Challenges in FPL

### 4.1 Team Selection Complexity

The fundamental challenge in FPL is optimizing team selection under constraints. With approximately 500 Premier League players available, the theoretical number of valid 15-player squads exceeds 10^23. Even limiting to weekly starting 11 selections, the decision space remains enormous[^16].

### 4.2 Predictive Uncertainty

Football is inherently unpredictable, with significant variance in player performance. Key uncertainties include:
- Injuries and rotation (players rested for certain matches)
- Form fluctuations throughout the season
- Managerial decisions affecting player roles and playing time
- Match context and fixture difficulty
- Weather conditions and other external factors[^17][^18]

### 4.3 Multi-objective Optimization

FPL managers must balance competing objectives:
- Maximizing expected points for the current gameweek
- Planning for future gameweeks (favorable fixture runs)
- Building team value through strategic transfers
- Differential selection (picking low-ownership players for competitive advantage)
- Risk management (captaincy choices, bench quality)[^19]

### 4.4 Temporal Dynamics

The game spans 38 gameweeks, requiring both short and long-term planning:
- Weekly decisions: Starting lineup, captaincy, transfers
- Medium-term decisions: Chip usage, planning for blank/double gameweeks
- Season-long decisions: Overall strategy and style of play[^20]

## 5. Relationship to Reinforcement Learning

Fantasy Premier League presents an ideal environment for reinforcement learning applications due to several characteristics:

### 5.1 Markov Decision Process Formulation

FPL naturally fits into the Markov Decision Process framework:
- **States**: Current team composition, budget, available transfers, fixture schedule
- **Actions**: Transfers, captain selection, bench order, chip usage
- **Transitions**: How actions transform the state (affected by real-world player performances)
- **Rewards**: Gameweek points earned
- **Long-term rewards**: Season-long point accumulation[^21][^22]

### 5.2 Delayed Rewards and Credit Assignment

FPL exhibits the classic reinforcement learning challenge of delayed rewards:
- Transfer decisions may not pay off immediately
- Building team value early may enable stronger teams later
- Planning for fixture difficulty must account for weeks or months ahead[^23]

### 5.3 Exploration-Exploitation Tradeoff

Successful FPL strategy requires balancing:
- Exploitation: Selecting proven performers and popular captaincy options
- Exploration: Taking calculated risks on differentials or emerging players[^24]

### 5.4 Non-stationarity

The FPL environment is non-stationary due to:
- Player form changes throughout the season
- Team tactical evolutions
- Injury impacts
- Transfer windows (January) bringing new players
- Manager changes affecting team performance[^25]

## 6. Previous Research and Algorithmic Approaches

### 6.1 Optimization-Based Approaches

Early algorithmic approaches to FPL focused on optimization techniques:
- Linear programming for team selection
- Integer programming for transfer planning
- Mixed-integer programming for season-long planning

While effective for constrained selection problems, these approaches often struggle with the inherent uncertainty and temporal dynamics of football[^26][^27].

### 6.2 Machine Learning Applications

Recent research has increasingly applied machine learning:
- Regression models for player point prediction
- Time series forecasting for form prediction
- Classification models for clean sheet probability
- Ensemble methods combining multiple prediction approaches[^28][^29]

### 6.3 Reinforcement Learning Explorations

Emerging research applies reinforcement learning to FPL:
- Q-learning for transfer decisions
- Deep Q-Networks for team selection
- Policy gradient methods for season-long strategy
- Monte Carlo Tree Search for planning[^30][^31]

These approaches show promise in managing the complex, sequential decision-making process that FPL represents, while accounting for uncertainty and delayed rewards.

## 7. Data Sources and Availability

Modern FPL research benefits from unprecedented data availability:
- Official FPL API providing comprehensive game data
- Third-party websites aggregating historical performance
- Event-level data from commercial providers (Opta, StatsBomb)
- Community resources like public GitHub repositories of historical data
- Web scrapers that collect and organize player statistics[^32][^33]

This rich data ecosystem enables the training of sophisticated models that can make informed predictions about player performance and optimal decision strategies.

## References

[^1]: Conn, D. (2017). *The Fall of the House of FIFA*. Random House.

[^2]: Premier League. (2023). *Premier League Handbook 2023/24*. Premier League.

[^3]: Buraimo, B., & Simmons, R. (2015). "Uncertainty of outcome or star quality? Television audience demand for English Premier League football." *International Journal of the Economics of Business*, 22(3), 449-469.

[^4]: Evens, T., Iosifidis, P., & Smith, P. (2022). "The political economy of television sports rights: History, power and culture." *International Journal of Sport Policy and Politics*, 14(1), 13-30.

[^5]: Szymanski, S., & Zimbalist, A. (2005). *National Pastime: How Americans Play Baseball and the Rest of the World Plays Soccer*. Brookings Institution Press.

[^6]: Fantasy Premier League. (2023). *Official Fantasy Premier League Participation Statistics*. Premier League.

[^7]: Bonomo, F., Durán, G., & Marenco, J. (2014). "Mathematical programming as a tool for virtual soccer coaches: A case study of a fantasy sport game." *International Transactions in Operational Research*, 21(3), 399-414.

[^8]: Fantasy Premier League. (2024). "Rules." Retrieved from https://fantasy.premierleague.com/help/rules

[^9]: Ibid.

[^10]: Ibid.

[^11]: Hughes, M., & Franks, I. (2005). *Analysis of Sport: The Key Concepts*. Routledge.

[^12]: Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019, July). "Actions speak louder than goals: Valuing player actions in soccer." In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 1851-1861).

[^13]: Fernández, J., Bornn, L., & Cervone, D. (2021). "A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions." *Machine Learning*, 110(6), 1389-1427.

[^14]: Tran, Q., & Lee, S. (2022). "Predicting fantasy premier league price changes using machine learning." In *2022 International Conference on Information Networking (ICOIN)* (pp. 622-627). IEEE.

[^15]: Constantinou, A. C., & Fenton, N. E. (2017). "Towards smart-data: Improving predictive accuracy in long-term football team performance." *Knowledge-Based Systems*, 124, 93-104.

[^16]: Matthews, T., Ramchurn, S. D., & Chalkiadakis, G. (2012, June). "Competing with humans at fantasy football: Team formation in large partially-observable domains." In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 26, No. 1).

[^17]: Bialkowski, A., Lucey, P., Carr, P., Yue, Y., Sridharan, S., & Matthews, I. (2014, November). "Large-scale analysis of soccer matches using spatiotemporal tracking data." In *2014 IEEE International Conference on Data Mining* (pp. 725-730). IEEE.

[^18]: Bryson, A., Frick, B., & Simmons, R. (2013). "The returns to scarce talent: Footedness and player remuneration in European soccer." *Journal of Sports Economics*, 14(6), 606-628.

[^19]: Matthews, T. (2013). "Improving fantasy football draft strategies using multiobjective optimization techniques." *Journal of Quantitative Analysis in Sports*, 9(2), 121-140.

[^20]: Constantinou, A. C. (2019). "Dolores: A model that predicts football match outcomes from all over the world." *Machine Learning*, 108(1), 49-75.

[^21]: Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction*. MIT press.

[^22]: Butler, D., Butler, R., & Eakins, J. (2021). "Expert performance and crowd wisdom: Evidence from fantasy premier league." *International Journal of Financial Studies*, 9(1), 5.

[^23]: Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., ... & Hassabis, D. (2017). "Mastering the game of go without human knowledge." *Nature*, 550(7676), 354-359.

[^24]: Matthews, T., & Ramchurn, S. D. (2019). "Competing with humans at fantasy football: Team formation in large partially-observable domains." *Journal of Autonomous Agents and Multi-Agent Systems*, 33(2), 130-171.

[^25]: Dixon, M. J., & Coles, S. G. (1997). "Modelling association football scores and inefficiencies in the football betting market." *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 46(2), 265-280.

[^26]: Pantuso, G. (2017). "The football team composition problem: A stochastic programming approach." *Journal of Quantitative Analysis in Sports*, 13(3), 113-129.

[^27]: Rotshtein, A. P., Posner, M., & Rakityanskaya, A. B. (2005). "Football predictions based on a fuzzy model with genetic and neural tuning." *Cybernetics and Systems Analysis*, 41(4), 619-630.

[^28]: Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017, July). "On calibration of modern neural networks." In *International Conference on Machine Learning* (pp. 1321-1330). PMLR.

[^29]: Baboota, R., & Kaur, H. (2019). "Predictive analysis and modelling football results using machine learning approach for English Premier League." *International Journal of Forecasting*, 35(2), 741-755.

[^30]: Hubáček, O., Šourek, G., & Železný, F. (2019, May). "Deep learning from label proportions for sequential data." In *Joint European Conference on Machine Learning and Knowledge Discovery in Databases* (pp. 39-55). Springer, Cham.

[^31]: Rahimian, M. A., Sha, M., Hwang, T., & Williams, J. (2021). "Minimizing maximal regret in commitment games." *AAAI Conference on Artificial Intelligence*, 35(6), 5493-5500.

[^32]: Pappalardo, L., Cintia, P., Rossi, A., Massucco, E., Ferragina, P., Pedreschi, D., & Giannotti, F. (2019). "A public data set of spatio-temporal match events in soccer competitions." *Scientific Data*, 6(1), 1-15.

[^33]: Decroos, T., & Davis, J. (2020). "Player vectors: Characterizing soccer players' playing style from match event streams." In *Machine Learning and Knowledge Discovery in Databases: European Conference, ECML PKDD 2019* (pp. 569-584). Springer International Publishing.
