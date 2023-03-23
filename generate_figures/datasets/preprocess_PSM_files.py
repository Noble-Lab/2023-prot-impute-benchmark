####################################################################
## Script to preprocess PSM files from CPTAC Study 47
## Author: Ayse Dincer 09/07/2021

# The script takes as input the PSM files downloaded from https://cptac-data-portal.georgetown.edu/study-summary/S047
# There are 11 folders containing PSM files where each TMT channel corresponds to a sample (total of 226 samples)
# The script also takes the sample mapping file to map each TMT channel to a sample id
# It also takes the peptide-level summary file made available in the protein report files to select only the filtered peptides listed in this file

# The script carries out the following preprocessing steps:
# For each PSM files, select only the peptides that occur in the filtered peptides list
# If there is more than one measurement mapped to the same peptide, select the measurement with the highest total ion intensity.
# If there is more than one PSM file measuring the same peptide, select the PSM file with the highest total ion intensity (TMT11-TotalAb).
# Gather the summed quantitations into a matrix where rows are peptides and columns are samples.

# The script records the results as a tab seperated file processed_peptide_quants.tsv and 
# also records the version of the dataset with the pool sample measurements for sanity check
####################################################################

import numpy as np
import pandas as pd

import re
from functools import reduce
import os

import time


#Define all dataset-specific parameters
peptide_filename = 'S047_Pediatric_Brain_Cancer/CBTTC_PBT_Proteome_CDAP_Protein_Report.r1/Gygi_TCMP_HMS_Proteome.peptides.tsv'
sample_mapping_filename = 'S047_HMS_Gygi_TMT11_Label_to_Sample_Mapping.tsv'
sample_mapping_file_keyword = 'Proteome Data Set Name'
main_foldername = 'S047_Pediatric_Brain_Cancer/'
sample_folder_keyword = 'CBTTC_PBT_Proteome_HMS'
eliminate_sample_keyword = 'Withdrawn'
pool_sample_keyword = 'Bridge'
pool_index = -1
list_of_channel_names = ['126', '127N', '127C', '128N', '128C', '129N', '129C', '130N', '130C', '131N', '131C']

#Read the list of peptides after filtering
peptide_df = pd.read_csv(peptide_filename, sep = '\t')
filtered_peptides = peptide_df['Peptide'].values
#print("Filtered peptides ", filtered_peptides)
print("Filtered peptides ", len(filtered_peptides), '\n')

#Read the TMT to sample mappings
sample_mapping_df = pd.read_csv(sample_mapping_filename, sep = '\t')
print("Sample mapping df ", sample_mapping_df.shape)
print("Sample mapping df ", sample_mapping_df, '\n')
folder_names = sample_mapping_df[sample_mapping_file_keyword].values

#Read the PSM files
sample_folders = os.listdir(main_foldername)
sample_folders = sorted(sample_folders)
print("\n\nReading PSM files...")

#Define the list of all peptides and dfs
all_peptides_across_all_folders = []
preprocessed_dfs = []

#Go over each sample directory
for sample_folder in sample_folders:
	if sample_folder_keyword in sample_folder: #make sure it is a sample folder
		print("---------------------")
		print("Folder: ", sample_folder)

		#Get names of all PSM files
		directory_name = main_foldername + sample_folder + '/' + sample_folder + '_PSM/'+ sample_folder + '_PSM.CAP.r1_tsv/'
		all_files = os.listdir(directory_name)
		all_files = sorted(all_files)
		print("Number of files ", len(all_files))
		#print("List of all files ", all_files)

		#Record all psm dfs, intensity dfs, all peptides
		all_psm_dfs = []
		all_intensity_dfs = []
		all_peptides_for_folder = []

		#Read PSM files for each sample
		for f in all_files:
			print("\n--------")
			print("Filename: ", f)
			psm_df = pd.read_csv(directory_name + f, sep = '\t')
			print("PSM df ", psm_df.shape)
			#print("PSM df ", psm_df)

			#Rename the peptides 
			print("Renaming peptides...")
			#Also record the original modified sequences
			all_peptide_names_with_mods = [ s for s in psm_df['PeptideSequence'].values]
			all_peptide_names = [ re.sub("[^a-zA-Z]+", "", str(s)) for s in psm_df['PeptideSequence'].values]
			psm_df['PeptideSequence'] = all_peptide_names
			psm_df['PeptideSequenceModifications'] = all_peptide_names_with_mods
			#print("PSM df ", psm_df['PeptideSequence'])

			#Add a key column to identify unique peptide ions
			psm_df['QueryCharge'] = psm_df['QueryCharge'].astype(str)
			psm_df['KeyValue'] = psm_df['PeptideSequenceModifications'].str.cat(psm_df['QueryCharge'],sep="-")
			
			#Select filtered peptides
			print("Selecting filtered peptides...")
			selected_peptides = np.intersect1d(psm_df['PeptideSequence'].values, filtered_peptides)
			psm_df  = psm_df[psm_df['PeptideSequence'].isin(filtered_peptides)]
			#print("Data df after filtering ", psm_df['PeptideSequence'])
			print("Data df after filtering ", psm_df.shape)
			
			#Define unique peptide ions
			unique_peptides = np.unique(psm_df['KeyValue'].values)
			all_peptides_for_folder.extend(unique_peptides)
			print("Number of unique peptides ions ", len(unique_peptides))

			#Map the TMT channels to samples
			print("Mapping the TMT channels to samples...")
			processed_peptides_list = []
			all_total_intensity_scores = []
			
			#Select the peptide ion with the max intensity
			for p in unique_peptides:

				#Focus on one peptide ion only
				sub_df = psm_df[psm_df['KeyValue'] == p]
				
				#If one peptide has multiple measurements, select the one with the highest TMT11-TotalAb
				selected_peptide = sub_df[sub_df['TMT11-TotalAb'] == sub_df['TMT11-TotalAb'].max()]
				psm_df = psm_df.drop(sub_df.index, axis = 0)

				#If multiple peptides have the same q_value, select randomly
				if selected_peptide.shape[0] > 1:
					selected_peptide = selected_peptide.sample()
				
				#Record values for each channel, also record the pool channel + total intensity
				selected_scores = selected_peptide[['TMT11-126C', 'TMT11-127N', 'TMT11-127C', 'TMT11-128N', 'TMT11-128C', 
								 					'TMT11-129N', 'TMT11-129C', 'TMT11-130N', 'TMT11-130C', 'TMT11-131N', 
								 					'TMT11-131C', 'TMT11-TotalAb']].values.ravel()
				selected_scores_list = []
				for s in selected_scores:
					if '/' not in str(s):
						selected_scores_list.append(s)
					else:
					 	selected_scores_list.append(float(str(s)[:str(s).index('/')]))

				sample_index = np.where(folder_names == sample_folder)
				new_sample_names = list(sample_mapping_df.iloc[sample_index[0][0]][list_of_channel_names].values.astype(str))
				new_sample_names[pool_index] = new_sample_names[pool_index].replace(' ', '') + '_' + sample_folder
				new_sample_names.append('TMT11-TotalAb_' + f)
				
				#define the final scores
				selected_scores = pd.DataFrame(np.array(selected_scores_list).reshape((1, -1)), index = selected_peptide.index, columns = new_sample_names)
				selected_peptide = pd.concat([selected_peptide[['PeptideSequence', 'PeptideSequenceModifications', 'Protein', 'QueryCharge', 'KeyValue']], selected_scores], axis = 1)
				selected_peptide.index = selected_peptide['KeyValue'].values
				
				total_intensity_scores = selected_peptide[['KeyValue', 'TMT11-TotalAb_' + f]]
				
				processed_peptides_list.append(selected_peptide)
				all_total_intensity_scores.append(total_intensity_scores)


			#Combine all peptides and intensities to define the new dataframe
			psm_df = pd.concat(processed_peptides_list)
			print("Processed df ", psm_df.shape)
			#print("Processed df ", psm_df)
			all_psm_dfs.append(psm_df)

			total_intensity_df = pd.concat(all_total_intensity_scores, axis = 0)
			print("Total intensity df ", total_intensity_df.shape)
			#print("Total intensity df ", total_intensity_df)
			all_intensity_dfs.append(total_intensity_df)

		print("\n\nCombining PSM files for sample...")
		#Record the unique peptides for the folder
		all_peptides_for_folder = np.unique(all_peptides_for_folder)
		all_peptides_across_all_folders.extend(all_peptides_for_folder)
		print("Total number of peptides for folder ", len(all_peptides_for_folder))
		
		#Combine intensity files
		intensity_df_for_folder = reduce(lambda x, y: pd.merge(x, y, on = ['KeyValue'],  how='outer').set_index('KeyValue'), all_intensity_dfs)
		intensity_df_for_folder_scores = intensity_df_for_folder[[s for s in intensity_df_for_folder.columns if ('TMT11-TotalAb' in s)]]
		print("Joined intensity df ", intensity_df_for_folder_scores.shape)
		#print("Joined intensity df ", intensity_df_for_folder_scores)

		#We now select the best PSM based on the total ion intensity for each peptide
		best_psm_index_for_each_peptide = np.nanargmax(np.array(intensity_df_for_folder_scores).astype(float), axis = 1)
		#print("Best indices for peptides ", best_psm_index_for_each_peptide)
		print("Best indices for peptides ", len(best_psm_index_for_each_peptide))
		
		#Combine all the selected psm_dfs
		all_selected_peptides = []
		for s in np.unique(best_psm_index_for_each_peptide):
			selected_indices = np.where(best_psm_index_for_each_peptide == s)[0]
			
			#Find the corresponding value
			sub_df = all_psm_dfs[s].loc[intensity_df_for_folder.index[selected_indices]]
			all_selected_peptides.append(sub_df)

		print("No of selected peptides ", len(all_selected_peptides))
		final_psm_df = pd.concat(all_selected_peptides, axis = 0)
		final_psm_df = final_psm_df.drop([i for i in final_psm_df.columns if 'TMT11-TotalAb' in i], axis = 1)
		preprocessed_dfs.append(final_psm_df)
		print("Final psm df  ", final_psm_df.shape)
		print("Final psm df  ", final_psm_df)		
		

# Combine all dfs and record each sample as a row
print("\n\nCombining all sample files...")

print("Number of all joined peptides ", len(np.unique(all_peptides_across_all_folders)))
final_df = reduce(lambda x, y: pd.merge(x, y, on = ['PeptideSequenceModifications', 'PeptideSequence', 'Protein', 'QueryCharge'],  how='outer'), preprocessed_dfs)
final_df = final_df.sort_values(by = ['PeptideSequence', 'PeptideSequenceModifications', 'QueryCharge'])
final_df.index = np.arange(final_df.shape[0]).astype(str)
final_df = final_df.drop([i for i in final_df.columns if eliminate_sample_keyword in i], axis = 1)
final_df = final_df.drop([i for i in final_df.columns if 'KeyValue' in i], axis = 1)
print("Final df ", final_df.shape)

#Find matching proteins and genes
final_df = final_df.drop(['Protein'], axis = 1)
final_df = pd.merge(final_df, peptide_df[['Peptide', 'Protein', 'Gene']], left_on='PeptideSequence', right_on='Peptide')
new_column_order = ['PeptideSequence', 'PeptideSequenceModifications', 'Protein', 'Gene', 'QueryCharge']
new_column_order.extend(final_df.columns[3:-3])
final_df = final_df[new_column_order]

#Save two versions, one with the pool samples for the sanity and the other without the pool samples
print("Final df with pool", final_df.shape)
print("Final df with pool", final_df)
final_df.to_csv('processed_peptide_quants_with_pools.tsv', sep = '\t')

final_df = final_df.drop([i for i in final_df.columns if pool_sample_keyword in i], axis = 1)
print("Final df ", final_df.shape)
print("Final df ", final_df)
final_df.to_csv('processed_peptide_quants.tsv', sep = '\t')

