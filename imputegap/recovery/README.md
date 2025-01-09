<img align="right" width="140" height="140" src="https://www.naterscreations.com/imputegap/logo_imputegab.png" >
<br /> <br />

# CONTAMINATION
## Patterns
<table>
    <tr>
        <td>M</td><td>Number of time series</td>
    </tr>
    <tr>
        <td>N</td><td>Lentgh of time series</td>
    </tr>
    <tr>
        <td>P</td><td>Starting position (protection)</td>
    </tr>
    <tr>
        <td>R</td><td>Missing rate of the pattern</td>
    </tr>
    <tr>
        <td>S</td><td>percentage of series selected</td>
    </tr>
    <tr>
        <td>W</td><td>Total number of values to remove</td>
    </tr>
    <tr>
        <td>B</td><td>Block size</td>
    </tr>
</table><br />

### MCAR
MCAR selects random series and remove block at random positions until a total of W of all points of time series are missing.
This pattern uses random number generator with fixed seed and will produce the same blocks every run.

<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>MAX</td>
    </tr>
    <tr>
        <td>M</td><td>MAX</td>
    </tr>
    <tr>
        <td>R</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>S</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>W</td><td>(N-P) * R</td>
    </tr>
    <tr>
        <td>B</td><td>2 - 20</td>
    </tr>
 </table>

<br />

### MISSING PERCENTAGE
**MISSING PERCENTAGE** selects a percentage of time series to contaminate, applying the desired percentage of missing values from the beginning to the end of each selected series.



<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>MAX</td>
    </tr>
    <tr>
        <td>M</td><td>MAX</td>
    </tr>
    <tr>
        <td>R</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>S</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>W</td><td>(N-P) * R</td>
    </tr>
    <tr>
        <td>B</td><td>R</td>
    </tr>
 </table><br />


### BLACKOUT
The **BLACKOUT** pattern introduces missing values across all time series by removing a specified percentage of data points from each series, creating uniform gaps for analysis.


<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>MAX</td>
    </tr>
    <tr>
        <td>M</td><td>MAX</td>
    </tr>
    <tr>
        <td>R</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>S</td><td>100%</td>
    </tr>
    <tr>
        <td>W</td><td>(N-P) * R</td>
    </tr>
    <tr>
        <td>B</td><td>R</td>
    </tr>
 </table><br />


### GAUSSIAN
The **GAUSSIAN** pattern introduces missing values into a percentage of time series, determined based on probabilities derived from a Gaussian distribution.

<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>MAX</td>
    </tr>
    <tr>
        <td>M</td><td>MAX</td>
    </tr>
    <tr>
        <td>R</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>S</td><td>100%</td>
    </tr>
    <tr>
        <td>W</td><td>(N-P) * R * probability</td>
    </tr>
    <tr>
        <td>B</td><td>R</td>
    </tr>
 </table><br />

### DISJOINT
The **DISJOINT** pattern introduces missing values into time series by selecting segments with non-overlapping intervals. This process continues until either the missing rate limit is reached or the series length is exhausted.

<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>MAX</td>
    </tr>
    <tr>
        <td>M</td><td>MAX</td>
    </tr>
    <tr>
        <td>R</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>S</td><td>100%</td>
    </tr>
    <tr>
        <td>W</td><td>(N-P) * R</td>
    </tr>
    <tr>
        <td>B</td><td>R</td>
    </tr>
 </table><br />

### OVERLAP
The **OVERLAP** pattern selects time series segments for introducing missing values by using a disjoint interval that is shifted by a specified percentage. This process continues until either the missing rate limit is reached or the series length is exhausted.


<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>MAX</td>
    </tr>
    <tr>
        <td>M</td><td>MAX</td>
    </tr>
    <tr>
        <td>R</td><td>1 - 100%</td>
    </tr>
    <tr>
        <td>S</td><td>100%</td>
    </tr>
    <tr>
        <td>W</td><td>(N-P) * R</td>
    </tr>
    <tr>
        <td>B</td><td>R</td>
    </tr>
 </table><br />