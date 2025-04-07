# Lab11_sars_cov_2_part3_mutationRate_and_variants

Viruses are not immune to mutations and evolution. During the pandemic, the SARS-CoV-2 virus mutated, evolved, and changed its characteristics, leaving us with various new strains and variants.

The evolution of new strains is tied to their rate of mutation. Therefore, we must know how fast they mutate to understand their evolution. Their mutation rate is meaningless without some references; thus, we will compare it to the rate of mutations of other viruses. Apart from the rate, the location of mutations is vital for evolving new characteristics while preserving their viability. Some specific mutations give rise to a new variant, and we will be interested in where those mutations happen in different variants. Lastly, we will focus on Slovenia and its variant landscape throughout the pandemic.

Let's log in and get our work set up.
```bash
mkdir lab_11
cd lab_11
```

# Part 1: Understanding Mutation Rate
How fast is the SARS-CoV-2 virus evolving? Can we empirically determine the speed of mutation? We can! Most biologists are very meticulous with their experiments and carefully document when samples are collected. We can then take a bunch of viral genomes, select a reference genome (usually the first known occurrence of the virus), and calculate the number of mutations from the reference to the remaining viral genomes. We can then use this information in conjunction with the sample collection dates to estimate how fast the viruses are evolving.

Let's load a file of variants. I have already done the courtesy of getting them in a file so there is no need to download.

```bash
cp /projects/class/binf3101_001/p1-sars-cov-2-variants.fasta ~/lab_11
```

## LQ 11.1a
There are ______ variants in our data set. There are _____ from China and _____ from Slovenia. 

## LQ 11.1b
TRUE or FALSE: All of the variants are the same length.

```python
module load python
python
import numpy as np          
import pandas as pd         # for saving classification in P4
from Bio import SeqIO       # for reading fasta files
```
Load the sequences from the .fasta file and save as a variable.

```python
# Load FASTA file (replace with your actual path)
sequences = list(SeqIO.parse("p1-sars-cov-2-variants.fasta", "fasta"))
```

You likely saw when inspecting the .fasta file the format of the description contains the accession number, country, and collection date. Observe how we are storing the data.
```python
# Extract metadata from descriptions
metadata = []
for seq in sequences:
    parts = seq.description.split('|')
    metadata.append({
        'accession': parts[0],
        'country': parts[1],
        'collection_date': pd.to_datetime(parts[2]),
        'sequence': seq
    })

# Create DataFrame and find earliest date
df = pd.DataFrame(metadata)
earliest_idx = df['collection_date'].idxmin()
patient_zero = df.loc[earliest_idx, 'sequence']
patient_zero_date = df.loc[earliest_idx, 'collection_date']
patient_zero_seq = str(df.loc[earliest_idx, 'sequence'])

print(f"Patient zero sequence collected on {patient_zero_date.date()}")
```

## LQ 11.2a
What is the pandas function in this loop used to convert collection dates into a Timestamp?

## LQ11.2b
What is the date Patient Zero was colleted? 

## LQ 11.2c
Inpect the patient_zero variable. What is the accession number of the earliest patient?

## LQ 11.2d
How does python deal with timestamps in which the day is not included in the date?

We are going to separate our earliest sequence from the rest so we can use this in an alignment.
```python
# Define file paths for output
patient_zero_file = "patient_zero_sarscov2.fasta"

# Write patient_zero sequence to file
with open(patient_zero_file, "w") as f_out:
    SeqIO.write(patient_zero, f_out, "fasta")

#Exit out of python, we now need to align our sequences.
exit()
```

```bash
cp /projects/class/binf3101_001/mafft_lab11.slurm ~/lab_11
```


Create the function to calculate this:
```python
import numpy as np

def jukes_cantor(reference_sequence: str, distant_sequence: str) -> float:
    \"\"\"The Jukes-Cantor correction for estimating genetic distances
    calculated with Hamming distance.

    Parameters
    ----------
    reference_sequence: str
        A string of nucleotides in a sequence used as a reference
        in an alignment with other (e.g. AGGT-GA)
    distant_sequence: str
        A string of nucleotides in a sequence after the alignment
        with a reference (e.g. AGC-AGA)

    Returns
    -------
    float
        The Jukes-Cantor corrected genetic distance using Hamming distance.
        For example 1.163.

    \"\"\"
    # Remove positions with indels ('-') in either sequence
    filtered_ref = []
    filtered_dist = []
    
    for ref, dist in zip(reference_sequence, distant_sequence):
        if ref != '-' and dist != '-':
            filtered_ref.append(ref)
            filtered_dist.append(dist)
    
    # Calculate the Hamming distance (proportion of differing sites)
    filtered_ref = ''.join(filtered_ref)
    filtered_dist = ''.join(filtered_dist)
    
    length = len(filtered_ref)
    if length == 0:
        raise ValueError("No valid positions to compare after filtering indels.")
    
    differences = sum(1 for a, b in zip(filtered_ref, filtered_dist) if a != b)
    p_distance = differences / length  # Proportion of differing sites
    
    # Apply Jukes-Cantor correction
    if p_distance >= 3/4:
        raise ValueError("p-distance is too high for Jukes-Cantor correction to be valid.")
    
    jc_distance = -3/4 * np.log(1 - (4/3) * p_distance)
    
    return jc_distance
"""

```

```python
sequences = list(SeqIO.parse("coronavirus_genome_alignment.fasta", "fasta"))
# Extract metadata from descriptions and create DataFrame
metadata = []
for seq in sequences:
    parts = seq.description.split('|')
    metadata.append({
        'accession': parts[0],
        'country': parts[1],
        'collection_date': pd.to_datetime(parts[2]),
        'sequence': seq.seq
    })

df = pd.DataFrame(metadata)

# Indexing our earliest patient again (now it is in alignment)
earliest_idx = df['collection_date'].idxmin()
patient_zero_seq = str(df.loc[earliest_idx, 'sequence'])
patient_zero_date = df.loc[earliest_idx, 'collection_date']

# Calculate JC corrected genetic distances to patient_zero
jc_distances = []
time_deltas = []


##a great way to keep track of an x,y axis
for _, row in df.iterrows():
    sequence = str(row['sequence'])
    collection_date = row['collection_date']
    
    # Calculate JC distance
    try:
        jc_distance = jukes_cantor(patient_zero_seq, sequence)
        jc_distances.append(jc_distance)
        
        # Calculate time elapsed in days from patient_zero's collection date
        time_elapsed = (collection_date - patient_zero_date).days
        time_deltas.append(time_elapsed)
    
    except ValueError as e:
        print(f"Skipping sequence {row['accession']} due to error: {e}")
        jc_distances.append(None)
        time_deltas.append(None)

# Remove None values for plotting
filtered_data = [(t, d) for t, d in zip(time_deltas, jc_distances) if t is not None and d is not None]
time_deltas_filtered, jc_distances_filtered = zip(*filtered_data)
```

Now plot the data
```python
# Plot JC corrected genetic distance vs. time elapsed
plt.figure(figsize=(10, 6))
plt.scatter(time_deltas_filtered, jc_distances_filtered, alpha=0.7, edgecolor='k')
plt.title("Jukes-Cantor Corrected Genetic Distance vs. Time Elapsed")
plt.xlabel("Time Elapsed (days)")
plt.ylabel("JC Corrected Genetic Distance")
plt.grid(True)

# Save the plot to a file
output_file = "jc_distance_vs_time_SARS-cov-2.png"  # change the file name to determine what 
plt.savefig(output_file, dpi=300)  # Save with high resolution (300 DPI)
print(f"Plot saved to {output_file}")
```

We've estimated the regression slope to SARS-CoV-2. Now what? Does the plot indicate a fast mutation rate? Or a slow mutation rate? Or an average mutation rate? We really can only tell with a frame of reference. In this exercise, we will look at two more viruses from recent outbreaks, the Zaire ebolavirus, and Zika virus, and determine their mutation rate. These will help us get a sufficient reference for the speed at which viruses mutate.

```bash
cp /projects/class/binf3101_001/p2-* ~/lab_11
```
Use ls to make sure they copied and so that you know the names of the files. You will need to edit your mafft files (or make copies and edit) to align these sequences. 

To make life easier, . You will still have to prepare the files for alignment with mafft. Make sure you are inputting the alignment file!! (not the fasta file you are copying)
```bash
cp /projects/class/binf3101_001/genetic_distance_plot.py
```
