# RICE13_pyomo
RICE model (version 2013 ) in Python (Pyomo)

This project aims at porting the [RICE model](https://en.wikipedia.org/wiki/DICE_model) (version 2013) by William Nordhaus in Pyomo, a Python package that emulates the behaviour of GAMS and other AMLs.
Differently from the DICE model, whose version 2013 has already been ported to Pyomo and can be found [here](https://github.com/moptimization/pythondice2013implementation), the RICE model do not consider the world as a whole, disaggregating instead it into 12 countries\regions. This complicates the analysis but allows to study the problem of international environmental agreements (IEAs) formation from a game theoretical (coalitional) point of view.

## How to...
The repository is self-contained and provides all the necessary data (in the folder _Data_) to run the model, together with the open-source non-linear solver IPOPT, version 3.9.1, in the folder _Solver_ (you may want to check if a newer version of the solver is available [here]( https://github.com/coin-or/Ipopt)). Data are taken from the Excell version of the RICE model 2013. Note that some Python packages are required to run the model, namely **Pyomo, Pandas, Numpy, Openpyxl, Math and Argparse**, together with their dependencies. 
The model can be run by launching the file RICE_2013.py from command line: 

…\RICE13_pyomo> python RICE_2013.py

Without providing any argument, the script will compute the solution of the model for the non-cooperative case (all countries\regions behaving fully egoistically) and for the fully cooperative case \[grand coalition\] (all countries\regions maximize the global wealth). This second option is equivalent to the DICE 2013 model. The time horizon is set to 14 time periods with a time step between each period of 10 years (the first time period is 2015 and, therefore, the model covers the temporal range from 2015 to 2155). By default, no intermediate coalitions are computed, where a coalition is a group of countries\regions that maximize their joint welfare instead of behaving egoistically (although they behave egoistically towards the non-members of the coalition). In order to activate the computation of coalitions and to change other default settings, appropriate options have to be passed.

### Options
| Option     | Type of value              | Description
|------------ |----------------------------|----------------------------------------------------|
|--T          |integer	                   |number of time periods (min = 2, max = 59, default = 15)|
|--tstep     |integer                    |number of years between each time period (valid values: 1, 2, 5, 10, 20; default = 10). If 20 is chosen as time step, the maximal number of time periods (T) is reduced to 29 such that T*tstep <= 590.|
|--tol	      |integer	                   |precision of the optimization algorithm (min = 7, max = 12, default = 7)|
|--max_it	  |integer	                   |maximum number of iterations performed by the optimization algorithm, (min = 500, max = 25000, default = 10000)|   
|--coop		    |string (no quotation marks) |Compute\do not compute the full cooperative solution (options are _True_ or _False_, default is _True_)|
|--nc		      |string (no quotation marks) |Compute\do not compute the fully non-cooperative solution (options are _True_ or _False_, default is _True_)|
|--coalition	|string		                   |Compute or not intermediate coalitions. The options are _none_ (no quotation marks) [no coalition is computed], _all_ (no quotation marks) [all coalitions are computed] or a list, inside double quotation marks, of countries\regions identifiers that constitute the coalition to be examined: e.g. "US, EU, LAM". The default is _none_. Note that passing the option _all_ will cause the computation of 4082 coalitions, computation that may take several hours (possibly days – a commercial solver instead of IPOPT may sensibly reduce the computation time)|

The identifiers of the countries\regions are the followings:

| Identifier | Country\Region |
|------------|----------------|
|US			|United States of America|
|JPN			|Japan|
|EU			|Western European Union|
|RUS		 | Russia|
|EUR		  |Eurasia|
|CHI			|China|
|IND			|India|
|MEST		|Middle East|
|AFR			|Africa|
|LAM		  |Latin America|
|OHI			|Other highly industrialized countries|
|OTH		  |Other South-East Asian countries|

Remember that all options should be provided without quotation marks, except for the eventual list of countries\regions that must be enclosed in DOUBLE quotation marks. Only one coalition a time (or all) can be selected.
Example of model run with parameters:

 …\RICE13_pyomo> python RICE_2013.py --T 25 --tstep 5 --tol 8 --max_iter 2000 --coop False --nc True --coalition "CHI, IND, AFR"
 
This will compute the model for the fully non-cooperative case and for the coalition including China, India and Africa. 25 time periods with five years time step (125 years) will be considered, while the optimization algorithm will have a precision till the eigth decimal place and will make a maximum of 2000 iterations.
 
Results are automatically exported in Excell format in the folder _Results_. The file _coop.xlsx_ retains the results for the fully cooperative case, the file _non_coop.xlsx_ the ones for the fully non-cooperative case, whereas, if a single coalition is selected, results are stored in the file named has the provided coalition: e.g. _CHI, IND, AFR.xslx_. If the option _all_ for -–coalition is selected, for each coalition will be created a file with increasing number: _coa1.xlsx_, _coa2.xlsx_,..., _coa4082.xlsx_. Note that, when this option is selected, the fully cooperative and the fully non-cooperative cases are computed by default, even if the argument _False_ is passed to --coop and to --nc. If a new simulation is run with modified parameters, it is opportune to move the results files currently in the _Results_ folder outside of it, otherwise they will be overwritten. 

### The results output
The results files have a sheet for each of the countries\regions displaying the values of all the model variables (rows) for each time period (columns). An additional sheet, called _global_, displays the variable that are common to all countries\regions. 
The countries\regions variables are:    

|Variable identifier|Description|
|------------|----------------|
|U		    |per period utility|
|K		    |capital stock|
|S		    |saving rate (as proportion of net output)|
|I		    |investments|
|Q		    |gross output|
|Y		    |net output (after environmental damages and abatement expenditures)|
|AB		  |abatement costs as proportion of gross output|
|D		    |environmental damages|
|C		    |consumption|
|E_ind		|industrial emissions (per country\region)|

The global variables are:

|Variable identifier|Description|
|------------|----------------|
|E_tot		|total industrial emissions at global level|
|M_at		|atmospheric GHG concentration|
|M_up		|biosphere and upper ocean GHG concentration|
|M_lo		|GHG concentration in lower strata of oceans|
|T_at		|atmospheric temperature change 	|
|T_lo		|temperature change in lower strata of oceans|
|F		    |radiative force|

In the files of results relative to coalitions there is an additional sheet, _members_, reporting the list of countries\regions member of the currently analyzed coalition and an indication of the number of iterations occurred to reach convergence. This last number, for the fully non-cooperative case, can be found in the _global_ sheet. Note that this is different from the iteration number of the optimization algorithm. For all cases, except the fully cooperative one, it is adopted the same algorithm proposed by Nordhaus, namely solving the optimization problem for each country at a time, fixing the values of the variables of all other countries as the results of the last optimization (or some starting values for the first optimization round). At every round, it is checked the difference of the control variables for all countries with the ones obtained in the previous round: if such difference is sufficiently small the algorithm is terminated since convergence has been reached, otherwise it is continued. The maximal number of iterations is fixed at 25. You should **always check this number** in the results file. If it is equal to 26 (the limit plus 1), it means that the algorithm has been interrupted before convergence has been reached, so the displayed results should not be trusted. 

### Analysis of coalition stability
Another information reported on the output file(s) is the result of the stability analysis of coalitions. This result is not reported for the fully non-cooperative case since it has scarce meaning. For all other cases, the following table is shown in the _global_ sheet:

|Type of stability| Result|
|------------ |----------------------------|
|Internallly Stable| True\False|
|Externally Stable| True\False. For the fully cooperative case, the _Non Applicable_ message is applied.|
|Fully Stable| True\False|
|PIS (Potential Internally Stable)| True\False|

Given a coalition, it is said to be internally stable if the payoff of each of its members is at least as much as the payoff that it would obtain by being the only one to leave the same coalition. Similarly, external stability requires that the payoff of each of the players outside of a coalition is at least as high as the payoff they could obtain by being the only one to join the same coalition. If both conditions are met simultaneously, the coalition is said to be fully stable, meaning that no player, either member or not of the coalition, has an incentive to unilaterally change its status (a concept similar to Nash equilibrium).  
Finally, potential internal stability (PIS) is a weaker condition of internal stability, requiring that the sum of the payoffs of the members of a coalition is higher than the sum of the payoffs they would obtain by abandoning, one at a time, the same coalition. In case a coalition fails to be internally stable, but meets the potential stability condition, this implies that there exist the possibility to stabilize (internally) the coalition through some transfers of payoffs among its members.
As a last remark, note that the players' payoffs discussed here are the sum of the utilities over all time periods for each player. 
