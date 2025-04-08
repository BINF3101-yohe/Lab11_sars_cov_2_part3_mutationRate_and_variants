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

```bash
module load python
python
```
In your python terminal, let's get ourselves sorted...

```python
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
Inspect the mafft file. Do not just run this blindly. Pay attention to the input because you will need to tweak this in the future. Make sure all of your inputs are in the lab_11 folder.

When you are sure everything is in its place. Run the mafft script.
```bash
sbatch mafft_lab11.slurm
```
The alignment should not take too long. We are going to calculate a mutation rate from this alignment. Remmber, by just taking the literal distances in the matrix ignores true evolutionary processes. 

When comparing two DNA sequences, you can count the number of nucleotide differences (substitutions). However, this observed difference underestimates the true evolutionary divergence because some positions may have undergone multiple substitutions over time, masking earlier changesIn the phylogenetics module, we talked very briefly about model selection. To more accurately infer the mutation rate, we are going to apply one of the most simple molecular models: Jukes-Cantor.

Over long evolutionary periods, some sites may experience multiple substitutions, making simple counting inaccurate. The Jukes-Cantor model accounts for these "hidden" substitutions. The Jukes-Cantor model is a mathematical tool used in molecular biology to estimate the evolutionary distance between two DNA sequences. It was developed by Charles Cantor and Thomas Jukes in 1969 and is one of the simplest models for nucleotide substitution in genetic sequences.

The Jukes-Cantor model assumes:
--All four nucleotides (A, T, C, G) are equally likely to substitute for one another.
--Substitution rates are constant across all sites in the sequence.
--Sites evolve independently of one another

Mathematically, this can be computed:

<img width="694" alt="Screenshot 2025-04-07 at 4 15 29 PM" src="https://github.com/user-attachments/assets/e9beb0b9-0821-48bf-9265-1da3b197da36" />

In matrix format, it can be visualized as the following:

<img width="1230" alt="Screenshot 2025-04-07 at 4 17 20 PM" src="https://github.com/user-attachments/assets/a3b4ef39-ae8d-43f4-acce-31a1a538461d" />


## LQ 11.3
A major limitation of J-C is that model assumes equal substitution rates. This makes it less accurate for highly divergent sequences or cases where these assumptions are violated. Give an example in which substitution rates may vary between base pairs (we talked about this in the phylogenetics mini-lab!

Below is a function to calculate the Jukes-Cantor corrected genetic distance from the reference sequence to all other and plot its dependence on the time elapsed from this starting point. Copy and paste into your python terminal. Have a look and see how the math is computed.
```python
def genetic_distance(reference_sequence: str, distant_sequence: str) -> float:
    """
    Calculates genetic distance using Jukes-Cantor correction or falls back to uncorrected p-distance.
    """
    # Remove positions with indels ('-') in either sequence
    filtered_ref = []
    filtered_dist = []
    
    for ref, dist in zip(reference_sequence, distant_sequence):
        if ref != '-' and dist != '-':
            filtered_ref.append(ref)
            filtered_dist.append(dist)
    
    filtered_ref = ''.join(filtered_ref)
    filtered_dist = ''.join(filtered_dist)
    
    length = len(filtered_ref)
    if length == 0:
        raise ValueError("No valid positions to compare after filtering indels.")
    
    differences = sum(1 for a, b in zip(filtered_ref, filtered_dist) if a != b)
    p_distance = differences / length  # Proportion of differing sites
    
    # Apply Jukes-Cantor correction if valid
    if p_distance < 3/4:
        jc_distance = -3/4 * np.log(1 - (4/3) * p_distance)
        return jc_distance
    else:
        # Fall back to uncorrected p-distance
        return p_distance

```

Your output from mafft_lab11.slurm should have created a file "coronavirus_genome_alignment.fasta". We are now going to comput the JC-distance from patient zero.

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

Now we are going to plot the data such that 

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
