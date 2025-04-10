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
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import pandas as pd

# Define the Jukes-Cantor correction function
def jukes_cantor(reference_sequence: str, distant_sequence: str) -> float:
    """
    The Jukes-Cantor correction for estimating genetic distances
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
    """
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
Using Cyberduck or filezilla (or your favorite file transfer), have a look at the beautiful plot you just made.

## LQ 11.4
The genetic distance of SARS-CoV-2 variants (increases/deacreases) as days pass. Just eyeballing, there is a (positive/negative/neutral) relationship with time and genetic distance. The slope that is fit amongst these points represents the (phylogenetic/mutation/alignment) rate for SARS-CoV-2.


We've estimated the regression slope to SARS-CoV-2. Now what? Does the plot indicate a fast mutation rate? Or a slow mutation rate? Or an average mutation rate? We really can only tell with a frame of reference. In this exercise, we will look at two more viruses from recent outbreaks, the Zaire ebolavirus, and Zika virus, and determine their mutation rate. These will help us get a sufficient reference for the speed at which viruses mutate.

Why Zika and ebola? SARS-CoV-2 is in a unique position where it is a worldwide phenomenon and warranted a global response in the past. As such, SARS-CoV-2 is most likely the most well-documented and tracked virus of all time. Even five years ago, sequencing on this scale would have been impossible. This creates a problem when we want to compare the mutation rate with other viruses. We need reference viruses that have gone through a similar lifecycle to SARS-CoV-2 and need to be recent enough such that sufficient sequencing data is available to estimate the slopes correctly. Unfortunately for us (but thankfully for humanity), only a handful of viruses fit this description (https://en.wikipedia.org/wiki/List_of_epidemics). Additionally, some developing countries still need the technological or economic capability to carry out this kind of sequencing on a large scale, making reliable data challenging to come by. We have chosen the Ebola virus and Zika virus, as their sequencing data is more or less reliable and plentiful enough.

```bash
cp /projects/class/binf3101_001/p2-* ~/lab_11
```
Use ls to make sure they copied and so that you know the names of the files. You will need to edit your mafft files (or make copies and edit) to align these sequences. 

To make life easier, we are going to . You will still have to prepare the files for alignment with mafft. Make sure you are inputting the alignment file!! (not the fasta file you are copying)
```bash
cp /projects/class/binf3101_001/genetic_distance_plot.py ~/lab_11
```

This python script is exaclty like the Jukes-Cantor function you ran above, except it includes the plotting fucntion within AND it outputs the slope for you. Let's see how it is run with our SARS-CoV-2 data.
You can now run the remaining scripts in bash.

```bash
python genetic_distance_plot.py coronavirus_genome_alignment.fasta --plot_label SARS_CoV2_Evolution
```

## LQ 11.5
Paste the regression coefficients for SARS-CoV-2.

Follow the following steps to answer the remaining lab questions.

1) Edit the mafft files and submit the slurms for zika and ebola.

2) Run the genetic_distance_plot.py for ebola and Zika

3) Upload your plots for Ebola and Zika. Analyze them in comparison to the plot from SARS-CoV-2


## LQ 11.6a
The initial outbreaks of Zika and Ebola that we are analyzing occurred in the same year.

## LQ 11.6b
Based on the calculated mutaiton rates, rank the three virsues in terms of slowest to fastest mutation rate.

## LQ 11.6c
Paste your plots for J-C distance v. time for Zika and Ebola

BONUS (10 points!) Plot the points and slopes of the three viruses on the same plot; color coded by virus. You can use chatGPT or your AI...or if you are THAT awesome, you can code it yourself. Paste your plot AND the code you used to plot this.

# Part 2: Variant-specific mutations and Identifying Variants

As the virus mutates, it inevitably evolves and proliferates around the world. Every so often, some mutations may prove especially beneficial to the spread of the virus, and this version of the virus spreads faster than other versions. When a version of a virus becomes especially prevalent inside a population, we call this a virus variant. Variants are nothing more than a naming scheme for viruses that have specific mutations. For instance, in Slovenia during 2022, we were dealing with the Omicron variant. Think of this as observing natural selection in real time. Some viruses have mutations that enable them to spread more easily throughout our population, which inevitably leads to the demise of other virus variants, which are not as good at proliferation. The result is survival of the fittest at the viral level, where, unfortunately, the fittest viruses seem to cause the most damage to us humans.

How do we identify variants? A variant is determined by several so-called defining mutations. Mutations can either be synonymous or nonsynonymous. Synonymous mutations are changes in nucleotide bases that result in the same encoded amino acid and are thus less important. Nonsynonymous mutations are nucleotide mutations that alter the amino acid sequence of a protein.

To determine mutations, we first have to select a reference genome, which we will say has no mutations. In most cases, this is the first known occurrence of the virus, but in our case, the reference NCBI genome from Wuhan in 2019 (NC_045512.2). Then, we align each viral genome of interest to this reference genome. All the differences between the reference genome and the genome of interest are said to be mutations.

To get a sense of the distribution of mutations across the genome in a variant, we will observe the most common mutations in Alpha (20I (Alpha, V1)) and Delta (21A (Delta)) variants. We will then try to answer whether the Delta variant emerged from the Alpha variant or evolved independently from a different strain of the virus.

```bash
cd lab_11
module load python
```

```python
from Bio import SeqIO

alpha_variants = [
    'EPI_ISL_2789189', 'EPI_ISL_2789042', 'EPI_ISL_1491060', 'EPI_ISL_1402029', 'EPI_ISL_6950370',
    'EPI_ISL_1625411', 'EPI_ISL_1335421', 'EPI_ISL_2644151', 'EPI_ISL_2982899', 'EPI_ISL_2644156',
    'EPI_ISL_2788965', 'EPI_ISL_2789059', 'EPI_ISL_2532608', 'EPI_ISL_2644516', 'EPI_ISL_2886579',
    'EPI_ISL_3316487', 'EPI_ISL_2886574', 'EPI_ISL_2532626', 'EPI_ISL_2886496', 'EPI_ISL_2492172',
    'EPI_ISL_2644108', 'EPI_ISL_1402024', 'EPI_ISL_2492224', 'EPI_ISL_2491984', 'EPI_ISL_2789018',
    'EPI_ISL_2886831', 'EPI_ISL_1491132', 'EPI_ISL_2492034', 'EPI_ISL_1266392', 'EPI_ISL_2983056'
]

delta_variants = [
    'EPI_ISL_3039380', 'EPI_ISL_4271386', 'EPI_ISL_5213082', 'EPI_ISL_3316705', 'EPI_ISL_3316997',
    'EPI_ISL_4251175', 'EPI_ISL_3471254', 'EPI_ISL_4271571', 'EPI_ISL_4270964', 'EPI_ISL_3317189',
    'EPI_ISL_3829145', 'EPI_ISL_3317102', 'EPI_ISL_4923915', 'EPI_ISL_3829384', 'EPI_ISL_4923898',
    'EPI_ISL_4270689', 'EPI_ISL_4270627', 'EPI_ISL_3828666', 'EPI_ISL_4253193', 'EPI_ISL_3828993',
    'EPI_ISL_3039412', 'EPI_ISL_4923029', 'EPI_ISL_4251446', 'EPI_ISL_4271300', 'EPI_ISL_4271597',
    'EPI_ISL_4271322', 'EPI_ISL_4922967', 'EPI_ISL_4251202', 'EPI_ISL_4251164', 'EPI_ISL_4270961',
    'EPI_ISL_4270530', 'EPI_ISL_4270924', 'EPI_ISL_3829530', 'EPI_ISL_3828321', 'EPI_ISL_4271408',
    'EPI_ISL_4271598', 'EPI_ISL_4924026', 'EPI_ISL_3316743'
]

reference_variant = [
    'NC_045512.2'
]

#Define input and output files
input_fasta="p1-sars-cov-2-variants.fasta"
alpha_output = "alpha_variants.fasta"
delta_output = "delta_variants.fasta"
reference_output="reference_variant.fasta"

# Function to extract sequences based on variant list
def extract_variants(input_file, output_file, variant_list):
    with open(output_file, "w") as outfile:
        for record in SeqIO.parse(input_file, "fasta"):
            if any(variant in record.description for variant in variant_list):
                SeqIO.write(record, outfile, "fasta")

# Usage# Extract Alpha variants
extract_variants(input_fasta, alpha_output, alpha_variants)

# Extract Delta variants
extract_variants(input_fasta, delta_output, delta_variants)

# Extract the reference patient zero
extract_variants(input_fasta, reference_output, reference_variant)
```

Exit out so we can use grep to count the nubmer of variants in each file.


## LQ 11.7
How many alpha and delta variants are in our data set?

Mutation rates can also vary between variants. Remember patient zero? This will serve as our reference sequence.
We are now going to calculate the nucleotide mismatches between each sequence and the reference sequence for the variant. The function below then computes the average mismatch ratio for each nucleotide position to generate an array of mutation frequencies ranging from 0 to 1. A value of 0 means no mutations occurred at that position in any sequence, while a value of 1 indicates that all sequences have a mutation at that position relative to the reference. Ensure that mutations are calculated separately for each variant.


```python
from Bio import SeqIO
import numpy as np

# Input FASTA files for Alpha and Delta variants
alpha_fasta = "alpha_variants.fasta"
delta_fasta = "delta_variants.fasta"

def read_reference_sequence(reference_fasta):
    record = next(SeqIO.parse(reference_fasta, "fasta"))  # Read the first record from the FASTA file
    return str(record.seq).upper()  # Return the sequence as an uppercase string

reference=read_reference_sequence("reference_variant.fasta")


def calculate_mismatch_frequencies(fasta_file, reference_sequence):
    sequences = []
    
    # Read sequences from the FASTA file
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    
    # Convert sequences to a NumPy array for easier manipulation
    seq_array = np.array([list(seq) for seq in sequences])
    
    # Ensure all sequences are aligned to the same length as the reference
    if seq_array.shape[1] != len(reference_sequence):
        raise ValueError("Sequences in the file do not match the length of the reference sequence.")
    
    # Convert the reference sequence to a NumPy array
    ref_array = np.array(list(reference_sequence))
    
    # Calculate mismatches (1 if mismatch, 0 if match)
    mismatches = seq_array != ref_array
    
    # Compute mutation frequencies (average mismatch ratio at each position)
    mutation_frequencies = mismatches.mean(axis=0)
    
    return mutation_frequencies

# Calculate mutation frequencies for Alpha and Delta variants
alpha_mutation_frequencies = calculate_mismatch_frequencies(alpha_fasta, reference)
delta_mutation_frequencies = calculate_mismatch_frequencies(delta_fasta, reference)

# Save results or print them
print("Alpha Mutation Frequencies:")
print(alpha_mutation_frequencies)

print("\nDelta Mutation Frequencies:")
print(delta_mutation_frequencies)

```
The printing should show an abbreviated matrix.

We will now the mutation occurrences across the whole genome for the Alpha and Delta variants separately on one figure (two subplots, one on top of another). We will represent the mutation occurance as a line plot =were x-axis is sequence index and y-axis is the average mutation occurance. We will show only the part of the genome above the 20000 nucleotides (focusing on the protein-coding genes) and mark locations of the SARS-CoV-2 genes (with colors and labels). We set the gene locations in the gene_locations variable. Expect a few sites with occurrence one and the rest close to zero.

```python
import matplotlib.pyplot as plt
import numpy as np

# Mutation frequencies for Alpha and Delta variants (calculated earlier)
alpha_mutation_frequencies = np.array(alpha_mutation_frequencies)
delta_mutation_frequencies = np.array(delta_mutation_frequencies)

# Gene locations
gene_locations = {
    'S': (21462, 25284),
    'E': (26144, 26372),
    'M': (26422, 27091),
    'N': (28173, 29433)
}

# Function to plot mutation occurrences
def plot_mutation_occurrences(alpha_freqs, delta_freqs, gene_locations, output_file):
    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # X-axis: nucleotide positions
    x_positions = np.arange(len(alpha_freqs))
    
    # Filter positions above 20,000 nucleotides
    mask = x_positions >= 20000
    x_filtered = x_positions[mask]
    alpha_filtered = alpha_freqs[mask]
    delta_filtered = delta_freqs[mask]
    
    # Plot Alpha variant mutation frequencies
    axes[0].plot(x_filtered, alpha_filtered, label="Alpha Variant", color="blue")
    axes[0].set_title("Alpha Variant Mutation Occurrences")
    axes[0].set_ylabel("Average Mutation Occurrence")
    
    # Plot Delta variant mutation frequencies
    axes[1].plot(x_filtered, delta_filtered, label="Delta Variant", color="red")
    axes[1].set_title("Delta Variant Mutation Occurrences")
    axes[1].set_xlabel("Nucleotide Position")
    axes[1].set_ylabel("Average Mutation Occurrence")
    
    # Mark gene locations on both subplots
    for gene_name, (start, end) in gene_locations.items():
        for ax in axes:
            ax.axvspan(start, end, alpha=0.3, label=f"{gene_name} Gene", color="grey")
            ax.text((start + end) / 2, ax.get_ylim()[1] * 0.8,
                    gene_name, color="black", ha="center", va="center", fontsize=10)
    
    # Add legends
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    
    # Adjust layout and save plot to file
    plt.tight_layout()
    plt.savefig(output_file)  # Save the figure to a file
    print(f"Plot saved to {output_file}")
    
    # Show the plot (optional)
    plt.show()

# Call the function to plot and save mutation occurrences
output_file = "mutation_occurrences.png"  # Change to your desired file name and format (e.g., .pdf)
plot_mutation_occurrences(alpha_mutation_frequencies, delta_mutation_frequencies, gene_locations, output_file)
```

Have a look at that plot! Wow!

## LQ11.8a
Paste your plot with the mutation occurrences in each variant.

We say a mutation is vital if its occurrence is higher than 0.5. Find all vital mutations in the spike gene ("S") and compare results between variants. 

## LQ11.8b
Inspect the number of vital mutations.


## LQ11.8c
There is one vital mutation that occurs in a protein-coding gene of the delta variants that does not occur in the alpha variants. What protein does this gene encode for and why is it important to SARS-CoV-2 function (hint: here is a good reference: https://www.nature.com/articles/s41467-022-32019-3).


There are a few sites where both variants mutated in the S-protein, but only one where the same mutation occurred.

To interpret the output, we report mutations from a reference like the following:
Mutation "G123A" means where G on position 123 in the reference mutates into an A.

Remember we are comparing to the reference!!

```python
# Extract S gene range
s_start, s_end = gene_locations['S']

# Function to find mutations in both variants
def find_mutations_in_both(alpha_fasta, delta_fasta, reference_sequence):
    mutations_in_both_variants = []
    same_mutation = []
    # Read sequences from FASTA files
    alpha_sequences = [str(record.seq).upper() for record in SeqIO.parse(alpha_fasta, "fasta")]
    delta_sequences = [str(record.seq).upper() for record in SeqIO.parse(delta_fasta, "fasta")]
    # Iterate over the S gene range
    for i in range(s_start - 1, s_end):  # Adjusting for 0-based indexing
        alpha_mutated_nucleotides = set(seq[i] for seq in alpha_sequences if seq[i] != reference_sequence[i])
        delta_mutated_nucleotides = set(seq[i] for seq in delta_sequences if seq[i] != reference_sequence[i])
        # Check if mutations occurred at this site in both variants
        if alpha_mutated_nucleotides and delta_mutated_nucleotides:
            mutations_in_both_variants.append(i + 1)  # Store the site index (convert back to 1-based indexing)
            # Check if the mutation is the same in both variants
            common_mutation = alpha_mutated_nucleotides.intersection(delta_mutated_nucleotides)
            if common_mutation:
                for mutation in common_mutation:
                    mutation_string = f"{reference_sequence[i]}{i + 1}{mutation}"
                    same_mutation.append(mutation_string)
    return mutations_in_both_variants, same_mutation

# Call the function with mutation frequencies and reference sequence
mutations_in_both_variants, same_mutation = find_mutations_in_both(
    alpha_fasta, delta_fasta, reference
)

# Print results
print("Mutations in Both Variants (Indices):", mutations_in_both_variants)
print("Same Mutations (Reference Nucleotide, Site, Variant Nucleotide):", same_mutation)
```
## LQ 11.9a
How many mutations occur in both variants at the same site?

## LQ 11.9b
How many of these are the same mutation?

## LQ 11.9c
 What are the two types of mutations you observe? 

 ## LQ 11.9d
Look at the values of the sites where the shared mutations occur. There are actually really only "two" shared mutations. What do I mean by this?

## LQ 11.9e
TRUE or FALSE: The indel mutation disrupts the protein-coding reading frame rendering the spike gene nonfunctional.

## LQ 11.9f
TRUE or FALSE: Shared mutations can evolve indepdently. 

## LQ 11.10
Inspect the plot you made and think about the resulting shared mutations. Did the Delta variant evolve from the Alpha variant? Explain your reasoning based on the common mutations.


