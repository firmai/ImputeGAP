<img align="right" width="70" height="70" src="https://www.naterscreations.com/imputegap/logo_imputegab.png" >
<br />

# CONTAMINATION
## Scenarios
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
        <td>R</td><td>Missing rate of the scenario</td>
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
This scenario uses random number generator with fixed seed and will produce the same blocks every run.

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
**MISSING PERCENTAGE** selects of percent of series to contaminate from the first to the last with a desired percentage of missing value to remove.

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
The **BLACKOUT** scenario selects all time series to introduce missing values. It removes a set percentage of data points from all series, creating gaps for further analysis.


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