![My Logo](../../assets/imputegab_logo.png)

# Scenarios
<table>
    <tr>
        <td>N</td><td>Lentgh of time series</td>
    </tr>
    <tr>
        <td>M</td><td>Number of time series</td>
    </tr>
    <tr>
        <td>R</td><td>Missing rate of the scenario</td>
    </tr>
    <tr>
        <td>W</td><td>Total number of values to remove</td>
    </tr>
</table>

### MCAR
MCAR removed from a random series at a random position until a total of W of all points of time series are missing.
This scenario uses random number generator with fixed seed and will produce the same blocks every run

<table>
    <tbody>Definition</tbody>
    <tr>
        <td>N</td><td>Selection by the user</td>
    </tr>
    <tr>
        <td>M</td><td>Max</td>
    </tr>
    <tr>
        <td>R</td><td>1 to 80%</td>
    </tr>
    <tr>
        <td>W</td><td>N * M * R</td>
    </tr>
    <tbody>Details</tbody>
    <tr>
        <td>Starting position</td><td>1 to 15% in the beginning of the series</td>
    </tr>
    <tr>
        <td>Missing blocks</td><td>1 to N-1</td>
    </tr>

 </table>