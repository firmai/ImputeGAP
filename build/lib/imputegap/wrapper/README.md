## Creation of your own CPP/C Library : 
<table>
    <tr>
        <td>requirements windows</td>
        <td>Armadillo, MinGW</td>
    </tr>
    <tr>
        <td>requirements linux/mac</td>
        <td>build-essential, g++, libarmadillo-dev</td>
    </tr>
</table>


#### Generation Linux/MAC :
```
g++ -fPIC -shared -o libcdrec.so ./Algorithms/CDMissingValueRecovery.cpp -larmadillo
```
#### Generation Windows :
```
g++ -fPIC -c ./Algorithms/CDMissingValueRecovery.cpp -I/mnt/c/armadillo/include

g++ -shared -o libcdrec.dll CDMissingValueRecovery.o -L/mnt/c/armadillo/lib -larmadillo
```