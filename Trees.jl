### A Pluto.jl notebook ###
# v0.17.7

using Markdown
using InteractiveUtils

# ╔═╡ 48706b3a-b446-11f0-263e-efa02e2f00be
using CSV, 
      DataFrames,
      PyPlot,
	  ScikitLearn, # machine learning package
	  StatsBase,
	  Random,
	  Statistics,
		PyCall,
		KernelDensity,
		Seaborn,
HypothesisTests
#	MLJ, #comment this out
#	NearestNeighborModels


# ╔═╡ 431bd934-b030-4aa2-8852-be426c0d0a6f
@sk_import neighbors: KNeighborsClassifier

# ╔═╡ b8d4de68-90a5-49fa-9587-699406b4d9d2
begin
	df_trees = CSV.read("alltrees_2025.csv", DataFrame)
	df_2024 = CSV.read("alltrees_2024.csv", DataFrame)
	df_2023 = CSV.read("alltrees_2023.csv", DataFrame)
	"reading in dfs"
end

# ╔═╡ 026c47f9-a8f8-48b9-acf4-94365d27aa77
begin
	function get_distance(child_ind, mama_ind, df)
		#child and mama as indices
		#df should be all trees for one island
		
		child_lat = df.latitude[child_ind]
		child_long = df.longitude[child_ind]
		mama_lat = df.latitude[mama_ind]
		mama_long = df.longitude[mama_ind]
	
		dist = sqrt(abs(child_lat - mama_lat)^2 + abs(child_long - mama_long)^2)
		#convert into meters
			#system is in decimal degrees
			#1 DD = 111 km
		
		return dist
	end
		
	function get_distance(tree1, tree2)
		dist = sqrt(abs(tree1.latitude - tree2.latitude)^2 + abs(tree1.longitude - tree2.longitude)^2)
		return dist
	end
end

# ╔═╡ 76c7835c-8386-4a45-9f9d-80a735f01a09
function plot_circle(x, y, r, axnum)
	axnum.add_patch(plt.Circle((x,y), r, edgecolor="grey", facecolor="none"))
end

# ╔═╡ 73524a92-aecb-46b7-8a5a-8a099f6c4e3b
begin

	#split of df_trees based on island
	bigisland_inds = findall([occursin("BI", i) for i in df_trees.name])
	BI_alltrees = DataFrame(
		:latitude => [df_trees.latitude[i] for i in bigisland_inds],
		:longitude => [df_trees.longitude[i] for i in bigisland_inds], 
		:name => [df_trees.name[i] for i in bigisland_inds])
	
	BI_kidtrees = copy(BI_alltrees)
	deleteat!(BI_kidtrees, findall([!occursin("Offspring", i) for i in BI_alltrees.name]))
	BI_adulttrees = copy(BI_alltrees)
	deleteat!(BI_adulttrees, findall([occursin("Offspring", i) for i in BI_alltrees.name]))
	
	O_alltrees = DataFrame(
		:latitude => [df_trees.latitude[j] for j in findall([occursin("O", i) for i in df_trees.name])],
		:longitude => [df_trees.longitude[j] for j in findall([occursin("O", i) for i in df_trees.name])],
		:name => [df_trees.name[j] for j in findall([occursin("O", i) for i in df_trees.name]) ]
	)

	deleteat!(O_alltrees, findall([occursin("BI", i) for i in O_alltrees.name]))
	deleteat!(O_alltrees, findall([occursin("K", i) for i in O_alltrees.name]))
	
	O_kidtrees = copy(O_alltrees)
	O_adulttrees = copy(O_alltrees)

	deleteat!(O_kidtrees, findall([!occursin("Offspring", i) for i in O_alltrees.name]))
	deleteat!(O_adulttrees, findall([occursin("Offspring", i) for i in O_alltrees.name]))
	
########################### Kaua'i trees ################################
	K_alltrees = DataFrame(
		:latitude => [df_trees.latitude[j] for j in findall([occursin("K", i) for i in df_trees.name])],
		:longitude => [df_trees.longitude[j] for j in findall([occursin("K", i) for i in df_trees.name])],
		:name => [df_trees.name[j] for j in findall([occursin("K", i) for i in df_trees.name])]
	)
	K_alltrees[!, :dists] = [Float64(0) for i in 1:length(K_alltrees.latitude)]

	K2_kid_ind = findall([i == "K2 Offspring" for i in K_alltrees.name])
	K5_kid_ind = findall([i == "K5 Offspring" for i in K_alltrees.name])
	K21_kid_ind = findall([occursin("K21 Offspring", i) for i in K_alltrees.name])
	
	K2_dists = [get_distance(i, findfirst(["K2" == j for j in K_alltrees.name]), K_alltrees) for i in K2_kid_ind]
	K5_dists = [get_distance(i, findfirst(["K5" == j for j in K_alltrees.name]), K_alltrees) for i in K5_kid_ind]
	K21_dists = [get_distance(i, findfirst(["K21" == j for j in K_alltrees.name]), K_alltrees) for i in K21_kid_ind]
	
	[K_alltrees.dists[j] = K2_dists[i] for (i,j) in enumerate(K2_kid_ind)]
	[K_alltrees.dists[j] = K5_dists[i] for (i,j) in enumerate(K5_kid_ind)]
	[K_alltrees.dists[j] = K21_dists[i] for (i,j) in enumerate(K21_kid_ind)]

	K_alltrees[!, :dists_m] = [i*111*1000 for i in K_alltrees.dists]
	
	K_kidtrees = copy(K_alltrees)
	K_adulttrees = copy(K_alltrees)
	
	deleteat!(K_kidtrees, findall([!occursin("Offspring", i) for i in K_alltrees.name]))
	deleteat!(K_kidtrees, findall([occursin("K3", i ) for i in K_kidtrees.name]))
		
	deleteat!(K_adulttrees, findall([occursin("Offspring", i) for i in K_alltrees.name]))

	deleteat!(K_adulttrees, findall([occursin("K3", i) for i in K_adulttrees.name]))
	deleteat!(K_adulttrees, findall([occursin("K4", i) for i in K_adulttrees.name]))
	deleteat!(K_adulttrees, findall([occursin("K80", i) for i in K_adulttrees.name]))

	K_adulttrees[!, :trees_in_circle] = [length(findall([i == "K21 Offspring" for i in K_alltrees.name])), length(findall([i == "K2 Offspring" for i in K_alltrees.name])), length(findall([i == "K5 Offspring" for i in K_alltrees.name]))]

#=	K_adulttrees[!, :num_kids] = vcat(
		length(findall([occursin("K21 Offspring", i) for i in K_kidtrees.name])),
		length(findall([occursin("K2 Offspring", i) for i in K_kidtrees.name])),
		length(findall([occursin("K5 Offspring", i) for i in K_kidtrees.name])))

	select!(K_adulttrees, Not(:dists))
	select!(K_adulttrees, Not(:dists_m))=#
	#########################
	"splitting up dataframes into island and adult/kid trees"
end

# ╔═╡ c47ac6a7-38e0-45b3-b9bd-79a8ab79ab98
begin
	hono_adulttrees = copy(O_adulttrees)
	Oahu_adulttrees_outside_Honolulu = []
	for (i, tree) in enumerate(eachrow(hono_adulttrees))
		if tree.latitude < 21.27 || tree.latitude > 21.33 || tree.longitude < -157.88 || tree.longitude > -157.8
			push!(Oahu_adulttrees_outside_Honolulu, i)
		end
	end
	deleteat!(hono_adulttrees, Oahu_adulttrees_outside_Honolulu)	
	
	hono_kidtrees = copy(O_kidtrees)
	Oahu_kidtrees_outside_Honolulu = []
	for (i, tree) in enumerate(eachrow(hono_kidtrees))
		if tree.latitude < 21.27 || tree.latitude > 21.33 || tree.longitude < -157.88 || tree.longitude > -157.8
			push!(Oahu_kidtrees_outside_Honolulu, i)
		end
	end
	
	#ylim(21.27, 21.33) #latitude of Honolulu
	#xlim(-157.88, -157.8) #longitude of Honolulu
	
	#the closest adult trees
	arggmin = [argmin([get_distance(hono_kidtrees[j,:], hono_adulttrees[i, :]) for i in 1:23]) for j in 1:419]
	
	df_allkidtreeinds_anddists = DataFrame(:adult_tree_ind => arggmin, :distance => [get_distance(O_kidtrees[i, :], O_adulttrees[j, :]) for (i,j) in enumerate(arggmin)])

	df_kidtrees_groupedby_closestadulttree = combine(groupby(df_allkidtreeinds_anddists, :adult_tree_ind), :distance .=> mean; renamecols= false)

	#converting distance in DD to meters
	df_kidtrees_groupedby_closestadulttree[!, :distance_m] = df_kidtrees_groupedby_closestadulttree.distance.*111000

	CSV.write("df_kidtrees_groupedby_closest_adult_tree.csv", df_kidtrees_groupedby_closestadulttree)
end

# ╔═╡ 364060ac-662d-47a6-8673-ac03bb65f321
begin
	figure()
	hist(df_kidtrees_groupedby_closestadulttree.distance_m)
	xlabel("Mean Distance of Offspring Trees to their Closest Adult Tree (m)")
	ylabel("Count")
	title("Offspring Tree Distances in Honolulu")
	gcf()
	savefig("kidtrees_groupedby_closestadulttree.png", dpi=500)
end

# ╔═╡ bbecb116-bf11-4d95-a78c-95fd6e0e2b39
begin#=
	adult_trees = O_adulttrees
	kid_trees = O_kidtrees
	toast = []
	ind_check = []
	
	for (i, tree) in enumerate(eachrow(adult_trees))
		tree_group = []
		lorp = 0
		for kid in eachrow(kid_trees)
			d = get_distance(tree, kid)
			
			if d <= R
		#just looking at the kid trees within the circle of this one adult tree
				push!(tree_group, kid)
			end
		end
		for (j, tree2) in enumerate(eachrow(adult_trees))
			if i <j && i != j
				for kid in tree_group
					if get_distance(tree2, kid) <= R
						lorp += 1 
						#push!()
					end
				end
			end
		end
		push!(toast, lorp)
end=#
	#=end
				for (j, adult_tree2) in enumerate(eachrow(adult_trees))
					#push!(ind_check, (i,j))
		#checking all other adult trees
					if i < j 
		#skipping duplicates of adult trees
						push!(ind_check, i)
						if get_distance(kid, adult_tree2) < R
		#is another adult tree within a radius of this kid tree?
							lorp += 1
						#	push!(toast, (get_distance(kid, adult_tree2)))
						end
					end
				end
		#	elseif d>R
		#		lorp = 0
			end
			
		end
		push!(toast, lorp)
	end=#
	"I think something to do with circles?"
end
			

# ╔═╡ 0e79c7fd-a60c-492e-9b9b-a1598f586959
begin
########################## Kaua'i Trees - 2023 #################
	K_alltrees_2023 = DataFrame(
		:latitude => [df_2023.latitude[j] for j in findall([occursin("K", i) for i in df_2023.name])],
		:longitude => [df_2023.longitude[j] for j in findall([occursin("K", i) for i in df_2023.name])],
		:name => [df_2023.name[j] for j in findall([occursin("K", i) for i in df_2023.name])]
	)
	K_alltrees_2023[!, :dists] = [Float64(0) for i in 1:length(K_alltrees_2023.latitude)]

	K2_kid_ind_2023 = findall([i == "K2 Offspring" for i in K_alltrees_2023.name])
	K5_kid_ind_2023 = findall([i == "K5 Offspring" for i in K_alltrees_2023.name])
	K21_kid_ind_2023 = findall([occursin("K21 Offspring", i) for i in K_alltrees_2023.name])
	
	K2_dists_2023 = [get_distance(i, findfirst(["K2" == j for j in K_alltrees_2023.name]), K_alltrees_2023) for i in K2_kid_ind_2023]
	K5_dists_2023 = [get_distance(i, findfirst(["K5" == j for j in K_alltrees_2023.name]), K_alltrees_2023) for i in K5_kid_ind_2023]
	K21_dists_2023 = [get_distance(i, findfirst(["K21" == j for j in K_alltrees_2023.name]), K_alltrees_2023) for i in K21_kid_ind_2023]
	
	[K_alltrees_2023.dists[j] = K2_dists_2023[i] for (i,j) in enumerate(K2_kid_ind_2023)]
	[K_alltrees_2023.dists[j] = K5_dists_2023[i] for (i,j) in enumerate(K5_kid_ind_2023)]
	[K_alltrees_2023.dists[j] = K21_dists_2023[i] for (i,j) in enumerate(K21_kid_ind_2023)]

	K_alltrees_2023[!, :dists_m] = [i*111*1000 for i in K_alltrees_2023.dists]
	
	K_kidtrees_2023 = copy(K_alltrees_2023)
	K_adulttrees_2023 = copy(K_alltrees_2023)
	
	deleteat!(K_kidtrees_2023, findall([!occursin("Offspring", i) for i in K_alltrees_2023.name]))
	deleteat!(K_kidtrees_2023, findall([occursin("K3", i ) for i in K_kidtrees_2023.name]))
		
	deleteat!(K_adulttrees_2023, findall([occursin("Offspring", i) for i in K_alltrees_2023.name]))

	deleteat!(K_adulttrees_2023, findall([occursin("K3", i) for i in K_adulttrees_2023.name]))
	deleteat!(K_adulttrees_2023, findall([occursin("K4", i) for i in K_adulttrees_2023.name]))
	deleteat!(K_adulttrees_2023, findall([occursin("K80", i) for i in K_adulttrees_2023.name]))

########################## Kaua'i Trees - 2024 #################

	K_alltrees_2024 = DataFrame(
		:latitude => [df_2024.latitude[j] for j in findall([occursin("K", i) for i in df_2024.name])],
		:longitude => [df_2024.longitude[j] for j in findall([occursin("K", i) for i in df_2024.name])],
		:name => [df_2024.name[j] for j in findall([occursin("K", i) for i in df_2024.name])]
	)
	K_alltrees_2024[!, :dists] = [Float64(0) for i in 1:length(K_alltrees_2024.latitude)]

	K2_kid_ind_2024 = findall([i == "K2 Offspring" for i in K_alltrees_2024.name])
	K5_kid_ind_2024 = findall([i == "K5 Offspring" for i in K_alltrees_2024.name])
	K21_kid_ind_2024 = findall([occursin("K21 Offspring", i) for i in K_alltrees_2024.name])
	
	K2_dists_2024 = [get_distance(i, findfirst(["K2" == j for j in K_alltrees_2024.name]), K_alltrees_2023) for i in K2_kid_ind_2024]
	K5_dists_2024 = [get_distance(i, findfirst(["K5" == j for j in K_alltrees_2024.name]), K_alltrees_2024) for i in K5_kid_ind_2024]
	K21_dists_2024 = [get_distance(i, findfirst(["K21" == j for j in K_alltrees_2024.name]), K_alltrees_2024) for i in K21_kid_ind_2024]
	
	[K_alltrees_2024.dists[j] = K2_dists_2024[i] for (i,j) in enumerate(K2_kid_ind_2024)]
	[K_alltrees_2024.dists[j] = K5_dists_2024[i] for (i,j) in enumerate(K5_kid_ind_2024)]
	[K_alltrees_2024.dists[j] = K21_dists_2024[i] for (i,j) in enumerate(K21_kid_ind_2024)]

	K_alltrees_2024[!, :dists_m] = [i*111*1000 for i in K_alltrees_2024.dists]
	
	K_kidtrees_2024 = copy(K_alltrees_2024)
	K_adulttrees_2024 = copy(K_alltrees_2024)
	
	deleteat!(K_kidtrees_2024, findall([!occursin("Offspring", i) for i in K_alltrees_2024.name]))
	deleteat!(K_kidtrees_2024, findall([occursin("K3", i ) for i in K_kidtrees_2024.name]))
		
	deleteat!(K_adulttrees_2024, findall([occursin("Offspring", i) for i in K_alltrees_2024.name]))

	deleteat!(K_adulttrees_2024, findall([occursin("K3", i) for i in K_adulttrees_2024.name]))
	deleteat!(K_adulttrees_2024, findall([occursin("K4", i) for i in K_adulttrees_2024.name]))
	deleteat!(K_adulttrees_2024, findall([occursin("K80", i) for i in K_adulttrees_2024.name]))


	"K trees - 2023 and 2024"
end

# ╔═╡ b513b08a-e041-4fa3-8e4d-21a2c79d9fda
begin
#	R = maximum(K_kidtrees.dists)
#	R_m = maximum(K_kidtrees.dists_m)
	
	#this is a value in meters from a lit search 11/24/26
	R_m = 17.84
	R = R_m/111000
	"defining radii"
end

# ╔═╡ 92bcaa09-0777-4365-b09f-82f5d137baba
function kid_trees_in_shared_space(offspring_trees, adult_trees)
	tree_count= []
	for tree in eachrow(offspring_trees)
		count_overlaps = 0
		for adulttree in eachrow(adult_trees)
			if get_distance(tree, adulttree) < R
				count_overlaps += 1
			end
			
		end
		push!(tree_count, count_overlaps)
	end
	return tree_count
	#will return number of overlapping adult tree circles that that kid tree is hanging out in
end

# ╔═╡ fcad27f6-c063-405b-a0b0-4730c0fead6c
begin
	figure()
	hist(filter(x -> x.name == "K21 Offspring", K_kidtrees).dists_m, color="blue", label="K21", alpha=0.5)
	hist(filter(x -> x.name == "K2 Offspring", K_kidtrees).dists_m, color="purple", label="K2", alpha=0.5)
	hist(filter(x -> x.name == "K5 Offspring", K_kidtrees).dists_m, color="orange", label="K5", alpha=0.5)
	axvline(R_m, color="purple")
	axvline(median(filter(x -> x.name == "K5 Offspring", K_kidtrees).dists_m), color="orange")
	axvline(median(filter(x -> x.name == "K21 Offspring", K_kidtrees).dists_m), color="blue")
	legend()
	
	gcf()
end
	
	

# ╔═╡ bf10ee1d-76e6-4a65-ae8d-df67ae6f724f
mutable struct Tree
	offspring::Bool
	latitude::Float64
	longitude::Float64
	name::String
	dists::Float64
	dists_m::Float64
end
	

# ╔═╡ 091dcb1e-bf15-442a-8b0f-eaec5c11fde0
begin
	# Define a sample struct
mutable struct MyStruct
    a::Int64
    b::Float64
end

# 1. Incorrect way (will cause issues)
 empty_array_any = []
 push!(empty_array_any, MyStruct(1, 2.0)) # This works but is not type-stable
#=
# 2. Correct way: Explicitly type the array
empty_array_typed = MyStruct[]

# Now you can push structs into the typed array
push!(empty_array_typed, MyStruct(1, 2.0))
push!(empty_array_typed, MyStruct(3, 4.0))=#

empty_array_any
# Output: 2-element Vector{MyStruct}:
#  MyStruct(1, 2.0)
#  MyStruct(3, 4.0)
end

# ╔═╡ 71a30053-7f0a-415d-8dab-c4095fdb6485
function get_median_dist(name_of_offspring, df, R_array)
	arr = [filter(x -> x.name == name_of_offspring && x.dists_m <= j, df).dists_m for j in R_array]
	arr2 = map(x -> length(x) == 0 ? 0 : x, arr)
	return [median(i) for i in arr2]
end

# ╔═╡ 09adf665-2e8a-41d5-bf9c-db3b6a40a4ef
function get_std_dev(name_of_offspring::String, df, R_array)
	arr = [filter(x -> x.name == name_of_offspring && x.dists_m <= j, df).dists_m for j in R_array]
#	arr = allowmissing(arr)
#	arr2 = map(x -> length(x) == 0 ? missing : x, arr)
	arr2 = map(x -> length(x) == 0 ? 0 : x, arr)
#	return map(x -> !ismissing(x) ? std(x) : x, arr2)
	return [std(i) for i in arr2]
end

# ╔═╡ ad2b5101-246a-4adf-ad52-c43685655b67
function get_density(name_of_offspring::String, df, R_array)
	count_lads = [count([i <= j for i in filter(x -> x.name == name_of_offspring, df).dists_m ]) for j in R_array]
	area = [π*R^2 for R in R_array]
	return count_lads ./ area
end

# ╔═╡ 862b5d5f-7a11-4219-9b2b-60c7d1fffb67
begin
	mess_df_21 = DataFrame(:R => [i for i in 0.5:0.5:R_m+0.5])
	mess_df_21[!, :ρ] = get_density("K21 Offspring", K_kidtrees, mess_df_21.R)
	mess_df_21[!, :median_d] = get_median_dist("K21 Offspring", K_kidtrees, mess_df_21.R)
	mess_df_21[!, :std_dev] = get_std_dev("K21 Offspring", K_kidtrees, mess_df_21.R)
	
	mess_df_2 = DataFrame(:R =>  [i for i in 0.5:0.5:R_m+0.5])
	mess_df_2[!, :ρ] = get_density("K2 Offspring", K_kidtrees, mess_df_2.R)
	mess_df_2[!, :median_d] = get_median_dist("K2 Offspring", K_kidtrees, mess_df_2.R)
	mess_df_2[!, :std_dev] = get_std_dev("K2 Offspring", K_kidtrees, mess_df_2.R)

	mess_df_5 = DataFrame(:R =>  [i for i in 0.5:0.5:R_m+0.5])
	mess_df_5[!, :ρ] = get_density("K5 Offspring", K_kidtrees, mess_df_2.R)
	mess_df_5[!, :median_d] = get_median_dist("K5 Offspring", K_kidtrees, mess_df_2.R)
	mess_df_5[!, :std_dev] = get_std_dev("K5 Offspring", K_kidtrees, mess_df_2.R)

	O_trees_combined = DataFrame(
	:latitude => vcat(O_adulttrees.latitude, O_kidtrees.latitude),
	:longitude => vcat(O_adulttrees.longitude, O_kidtrees.longitude),
	:names => vcat(O_adulttrees.name, O_kidtrees.name))
	"messy dfs for K and O"
end

# ╔═╡ 5bcd0997-1fe7-4f5b-95b6-56e7a8822217
begin
	R_array = mess_df_21.R 
	x = K_kidtrees
	df=K_kidtrees
	trees_in_R = [filter(x -> x.name == "K21 Offspring" && x.dists_m <= j, df) for j in R_array]
	yarr= []
	for k in trees_in_R
		med_arr = []
		for (i, m) in enumerate(eachrow(k))
			for (j, n) in enumerate(eachrow(k))
				if i < j
					
			#		push!(yarr, (median(get_distance(i, j)), get_distance(i, j)))
					push!(yarr, get_distance(m, n))
				end
				push!(med_arr, size(yarr))
			end
		end
		
	#	push!(yarr, median(med_arr))
	end

	
	
end

# ╔═╡ cb6808ce-07b3-4fc7-ac76-fd7559082347
begin
	f80, (ax80, ax81, ax82) = subplots(3,1)
	ax83 = ax80.twinx()
	ax84=ax81.twinx()
	ax85=ax82.twinx()
	
	f80.tight_layout()
	subplots_adjust(hspace=0)

	ax81.set_xticks([])
	ax80.set_xticks([])
	
	ax80.plot(mess_df_21.R, mess_df_21.ρ, color="red")
	ax83.plot(mess_df_21.R, mess_df_21.median_d, label="K21", color="green")
	ax80.axvline(110, color="black", alpha=0.3)
	ax80.set_xlim(0,330)
	
	ax81.plot(mess_df_2.R, mess_df_2.ρ, color="red")
	ax84.plot(mess_df_2.R, mess_df_2.median_d,  label="K2", color="green")
	ax81.axvline(17.5, color="black", alpha=0.3)
	ax81.set_xlim(0,330)
	
	ax82.plot(mess_df_5.R, mess_df_5.ρ, color="red")
	ax85.plot(mess_df_5.R, mess_df_5.median_d, color="green")
	ax82.axvline(57, color="black", alpha=0.3)
	ax82.set_xlim(0,330)

	ax82.set_xlabel("Radius of Parent Circle (m)", size=15)
	ax81.set_ylabel(L"Density \;of \;Offspring \;(\frac{trees}{m^2})", color="red", size=15)
	ax84.set_ylabel("Median Distance to Parent Tree (m)", rotation = 270, labelpad=20, color="green", size=15)	

	gcf()
	#savefig("scanning_for_radius.png", dpi=500)
end

# ╔═╡ 52ff9693-f2c2-42e3-9211-02b66df20202
begin
#=	f2, (ax2) = subplots(figsize=(8,8))
	ax2.scatter( K_kidtrees.longitude, K_kidtrees.latitude, label="offspring", color="orange")
	ax2.scatter(K_adulttrees.longitude,  K_adulttrees.latitude, label="adult", color="blue", s=120)
	[plot_circle(K_adulttrees.longitude[i], K_adulttrees.latitude[i], R, ax2) for i in 1:3]
#1 DD = 111 km
#	[ax2.text(K_adulttrees.longitude[i], K_adulttrees.latitude[i],  K_adulttrees.name[i]) for i in 1:3]
	ax2.set_xlabel("Longitude (degrees)", fontsize=15)
	ax2.set_ylabel("Latitude (degrees)", fontsize=15)
	title("Kaua'i Trees and Offspring ", fontsize=20)
	
#	ylim(21.9555, 21.983)
#	xlim(-159.725, -159.7)
	#K5
#	ylim(22.208,22.216)
#	xlim(-159.3875, -159.38)

	#K21
#	ylim(21.964,21.972)
#	xlim(-159.716, -159.708)

	#K2
#	ylim(21.972,21.98)
#	xlim(-159.363, -159.371)

	gcf()
#	savefig("Kauai_trees_5_R615.png", dpi=500)=#
	"Lat Long Kaua'i trees"
end

# ╔═╡ 32a4f1f2-cc4f-4c95-902f-f08056373046
begin
	figure()
	hist(K_kidtrees.dists_m, bins=20)
	axvline(median([K_kidtrees.dists_m[j] for j in findall([i <400 for i in K_kidtrees.dists_m])]), color="red", label="median (disregarding points > 400 m)")
	axvline(median(K_kidtrees.dists_m), color="pink", label="median (all points)")
	ylabel("Number of Trees")
	xlabel("Distance to Parent Tree (m)")
	title("Kaua'i Offspring-Parent Distances", fontsize=20)
	legend()
	gcf()
	"Kaua'i Offspring-Parent Distances"
end

# ╔═╡ 60cb902c-6648-4708-b070-12622d868fea
begin
	figure()
	bar(1, count(isequal("K2 Offspring"), K_kidtrees.name), label="K2 Offspring")
	bar(2, length(findall([occursin("K3 Offspring", i) for i in K_kidtrees.name])), label = "K3 Offspring")
	bar(3, count(isequal("K4 Offspring"), K_kidtrees.name), label="K4 Offspring")
	bar(4, count(isequal("K5 Offspring"), K_kidtrees.name), label= "K5 Offspring")
	bar(5, count(isequal("K20 Offspring"), K_kidtrees.name), label="K20 Offspring")
	bar(6, length(findall([occursin("K21 Offspring", i) for i in K_kidtrees.name])), label="K21 Offspring")
	bar(7, count(isequal("K73 Offspring"), K_kidtrees.name), label="K73 Offspring")
	bar(8, count(isequal("K80 Offspring"), K_kidtrees.name), label="K80 Offspring")
	xticks([i for i in 1:8], ["K2", "K3", "K4", "K5", "K20", "K21", "K73","K80"])
	xlabel("Parent Tree")
	ylabel("Offspring Count")
	title("Offspring Trees per Parent Tree - Kaua'i")
	gcf()
	"Offspring Trees per Parent Tree - Kaua'i"
end

# ╔═╡ 3b3162b8-46c7-47da-88e3-e103197436ae
begin
	@pyimport matplotlib as mpl
	@pyimport matplotlib.patches as ptc
	#@pyimport geopandas as gpd
#	@pyimport cartopy.crs as ccrs
end

# ╔═╡ 5c147595-ec5d-4b82-8b52-e129e8ce75dc
begin
	function too_close(tree_i, tree_j)
       		hypotenuse = sqrt((tree_i.latitude - tree_j.latitude)^2 + (tree_i.longitude - tree_j.longitude)^2)
        return hypotenuse < 2*R
    end

   function overlapping_trees(proposed_tree, planted_trees::DataFrame)
		overlapping_circles = 0
        for i in eachrow(planted_trees)
			#check to make sure proposed tree isn't i
			if i != proposed_tree
	            if too_close(proposed_tree, i)
					overlapping_circles += 1
	           end
			end
        end
        return overlapping_circles
	
  end


end

# ╔═╡ 50fadefb-6cba-4a00-91ae-5d6eea604d0c
begin
	figure()
	hist(O_adulttrees.density_meth1, label="Method 2: Double Counting", alpha=0.5)
	hist(O_adulttrees.density_meth2, label="Method 2: Exclusion", alpha=0.5)
	hist(O_adulttrees.density_meth3, label="Method 2: Sharing", alpha=0.5)
	
	xlabel(L"Density \; (\frac{trees}{m^2})", size=15)
	title("Oahu - Different Density Ideas, R=17.84 m", size=15)
	legend()
	gcf()
end

# ╔═╡ c04ec5c0-fd1c-4eb5-a232-4dc9f04feb04
function get_trees_in_circle(parent_tree, df_kidtrees)
	circle_trees = 0
	for kid_tree in eachrow(df_kidtrees)
		d = get_distance(parent_tree, kid_tree)
		if d <= R
			circle_trees += 1.0
		end
	end
	return circle_trees
end

# ╔═╡ 3c7b52dc-211d-485b-898b-94257cdc5951
begin
	O_adulttrees[!, :density_meth1] = O_adulttrees.trees_in_circle./(π*R_m^2)

	#which O_kidtrees are in overlapping circles?
	#it should be ones where the number of nearby adults ("nearby" meaning within R) is more than 1
	#[i > 1 for i in O_kidtrees.num_near_adults] yields only one tree, at index 394
	#tree 394 is only tree in multiple circles (2) 

	#recount O_adulttrees.trees_in_circle, discarding 394
	O_kidtrees_meth2 = copy(O_kidtrees)
	delete!(O_kidtrees_meth2, 2)
	O_adulttrees[!, :trees_in_circle_meth2] = [get_trees_in_circle(i, O_kidtrees_meth2) for i in eachrow(O_adulttrees)]
	O_adulttrees[!, :density_meth2] = O_adulttrees.trees_in_circle_meth2./(π*R_m^2)
	
	#eh I'm just gonna change adult trees 27 and 28 because they're the ones with tree 394	
	O_adulttrees[!, :density_meth3] = (push!(O_adulttrees.trees_in_circle[1:26], 0.5, 0.5, O_adulttrees.trees_in_circle[29]))./(π*R_m^2)
end

# ╔═╡ 5ea415db-4073-4b70-9180-332b7579ada0
begin
	#=
f3, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = subplots(3,3, figsize=(8,8), sharex=true, sharey=true)
#=	ax1.set_yticks([])
	ax2.set_yticks([])
	ax3.set_yticks([])
	ax4.set_yticks([])
	ax5.set_yticks([])
	ax6.set_yticks([])
	ax7.set_yticks([])
	ax8.set_yticks([])
	ax9.set_yticks([])=#

	
	ax1.set_title("K5", size=15)
	ax4.set_title("K21", size=15)
	ax7.set_title("K2", size=15)
	ax1.set_ylabel("2023", size=15)
	ax2.set_ylabel("2024", size=15)
	ax3.set_ylabel("2025", size=15)

	ax1.hist([K_kidtrees_2023.dists_m[i] for i in findall([occursin("K5 Offspring", i) for i in K_kidtrees_2023.name])])
	ax1.axvline(median([K_kidtrees_2023.dists_m[i] for i in findall([occursin("K5 Offspring", i) for i in K_kidtrees_2023.name])]), color="purple", linestyle="--")
	ax2.hist([K_kidtrees_2024.dists_m[i] for i in findall([occursin("K5 Offspring", i) for i in K_kidtrees_2024.name])])
	ax2.axvline(median([K_kidtrees_2024.dists_m[i] for i in findall([occursin("K5 Offspring", i) for i in K_kidtrees_2024.name])]), color="purple", linestyle="--")
	ax3.hist([K_kidtrees.dists_m[i] for i in findall([occursin("K5 Offspring", i) for i in K_kidtrees.name])])
	ax3.axvline(median([K_kidtrees.dists_m[i] for i in findall([occursin("K5 Offspring", i) for i in K_kidtrees.name])]), color="purple", linestyle="--")
	
	ax4.hist([K_kidtrees_2023.dists_m[i] for i in findall([occursin("K21 Offspring", i) for i in K_kidtrees_2023.name])])
	ax4.axvline(median([K_kidtrees_2023.dists_m[i] for i in findall([occursin("K21 Offspring", i) for i in K_kidtrees_2023.name])]), color="purple", linestyle="--")
	ax5.hist([K_kidtrees_2024.dists_m[i] for i in findall([occursin("K21 Offspring", i) for i in K_kidtrees_2024.name])])
	ax5.axvline(median([K_kidtrees_2024.dists_m[i] for i in findall([occursin("K21 Offspring", i) for i in K_kidtrees_2024.name])]), color="purple", linestyle="--")
	ax6.hist([K_kidtrees.dists_m[i] for i in findall([occursin("K21 Offspring", i) for i in K_kidtrees.name])])
	ax6.axvline(median([K_kidtrees.dists_m[i] for i in findall([occursin("K21 Offspring", i) for i in K_kidtrees.name])]), color="purple", linestyle="--")

	ax7.hist([K_kidtrees_2023.dists_m[i] for i in findall([occursin("K2 Offspring", i) for i in K_kidtrees_2023.name])])
	ax7.axvline(median([K_kidtrees_2023.dists_m[i] for i in findall([occursin("K2 Offspring", i) for i in K_kidtrees_2023.name])]), color="purple", linestyle="--")
	ax8.hist([K_kidtrees_2024.dists_m[i] for i in findall([occursin("K2 Offspring", i) for i in K_kidtrees_2024.name])])
	ax8.axvline(median([K_kidtrees_2024.dists_m[i] for i in findall([occursin("K2 Offspring", i) for i in K_kidtrees_2024.name])]), color="purple", linestyle="--")
	ax9.hist([K_kidtrees.dists_m[i] for i in findall([occursin("K2 Offspring", i) for i in K_kidtrees.name])])
	ax9.axvline(median([K_kidtrees.dists_m[i] for i in findall([occursin("K2 Offspring", i) for i in K_kidtrees.name])]), color="purple", linestyle="--")
	
	ax6.set_xlabel("Distance from Parent Tree (m)",size=20)
	gcf()=#
	"Kauai trees by time"
#	savefig("Kauai_trees_by_year.png", dpi=500)
end
	

# ╔═╡ 82c2344c-6a07-4187-bb30-6ac7918080cc
begin
	#=figure()
	plot((1,2,3), (
		length(findall([occursin("K5 Offspring", i) for i in K_kidtrees_2023.name])),
		length(findall([occursin("K5 Offspring", i) for i in K_kidtrees_2024.name])),
		length(findall([occursin("K5 Offspring", i) for i in K_kidtrees.name]))),
		color="blue", marker="o", label="K5")
	plot((1,2,3), (
		length(findall([occursin("K21 Offspring", i) for i in K_kidtrees_2023.name])),
		length(findall([occursin("K21 Offspring", i) for i in K_kidtrees_2024.name])),
		length(findall([occursin("K21 Offspring", i) for i in K_kidtrees.name]))),
		color="red", marker="o", label="K21")
	plot((1,2,3), (
		length(findall([occursin("K2 Offspring", i) for i in K_kidtrees_2023.name])),
		length(findall([occursin("K2 Offspring", i) for i in K_kidtrees_2024.name])),
		length(findall([occursin("K2 Offspring", i) for i in K_kidtrees.name]))),
		color="lightgreen", marker="o", label="K2")
	ylabel("Number of Offspring Trees", size=14)
	xticks((1,2,3), ("2023", "2024", "2025"))
	xlabel("Sampling Years", size=14)
	title("Kaua'i Tree Offspring over Time", size=18)
	legend()
	gcf()=#
#	savefig("Kauai_trees_over_time.png", dpi=500)
	"Kaua'i Tree Offspring over Time - Scatter"
end

# ╔═╡ 2374a2a9-97ce-4bd7-8501-1c2fa4cb6b2e
begin#=
f3, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = subplots(3,3, figsize=(8,8), sharey=true)
#=	ax1.set_yticks([])
	ax2.set_yticks([])
	ax3.set_yticks([])
	ax4.set_yticks([])
	ax5.set_yticks([])
	ax6.set_yticks([])
	ax7.set_yticks([])
	ax8.set_yticks([])
	ax9.set_yticks([])=#

#	f3.supxlabel("Distance from Parent Tree (m)", size=20)
	
	ax1.set_title("K5", size=15)
	ax4.set_title("K21", size=15)
	ax7.set_title("K2", size=15)
	ax1.set_ylabel("2023", size=15)
	ax2.set_ylabel("2024", size=15)
	ax3.set_ylabel("2025", size=15)

	ax1.boxplot([K_kidtrees_2023.dists_m[i] for i in findall([occursin("K5 Offspring", i) for i in K_kidtrees_2023.name])])
	ax2.boxplot([K_kidtrees_2024.dists_m[i] for i in findall([occursin("K5 Offspring", i) for i in K_kidtrees_2024.name])])
	ax3.boxplot([K_kidtrees.dists_m[i] for i in findall([occursin("K5 Offspring", i) for i in K_kidtrees.name])])
	
	ax4.boxplot([K_kidtrees_2023.dists_m[i] for i in findall([occursin("K21 Offspring", i) for i in K_kidtrees_2023.name])])
	ax5.boxplot([K_kidtrees_2024.dists_m[i] for i in findall([occursin("K21 Offspring", i) for i in K_kidtrees_2024.name])])
	ax6.boxplot([K_kidtrees.dists_m[i] for i in findall([occursin("K21 Offspring", i) for i in K_kidtrees.name])])

	ax7.boxplot([K_kidtrees_2023.dists_m[i] for i in findall([occursin("K2 Offspring", i) for i in K_kidtrees_2023.name])])
	ax8.boxplot([K_kidtrees_2024.dists_m[i] for i in findall([occursin("K2 Offspring", i) for i in K_kidtrees_2024.name])])
	ax9.boxplot([K_kidtrees.dists_m[i] for i in findall([occursin("K2 Offspring", i) for i in K_kidtrees.name])])
	
	ax6.set_xlabel("Distance from Parent Tree (m)", size=20)
	gcf()=#
end

# ╔═╡ c28ead14-a09b-4641-aa45-a44488278248
begin
	f, (ax) = subplots(1,1, figsize=(10,8))
	
	scatter(O_adulttrees.longitude, O_adulttrees.latitude)
	scatter(O_adulttrees.longitude[26], O_adulttrees.latitude[26], color="green")
	scatter(O_adulttrees.longitude[27], O_adulttrees.latitude[27], color="purple")
	scatter(O_adulttrees.longitude[28], O_adulttrees.latitude[28], color="yellow")
	scatter(O_adulttrees.longitude[29], O_adulttrees.latitude[29], color="orange")
	scatter(O_kidtrees.longitude[394], O_kidtrees.latitude[394], color="red")

	#scatter(O_adulttrees.longitude[1], O_adulttrees.latitude[1], label="Adult", color="red")
#	[plot_circle(O_trees_combined.longitude[i], O_trees_combined.latitude[i], R_m/(1000*110), ax) for i in 1:length(O_trees_combined.latitude)]
	[plot_circle(O_adulttrees.longitude[i], O_adulttrees.latitude[i], R, ax) for i in findall([too_close(O_kidtrees[394,:], O_adulttrees[i, :]) for i in 1:29])]
#	[ax.text(O_adulttrees.latitude[i], O_adulttrees.longitude[i], O_adulttrees.name[i]) for i in 1:length(O_adulttrees.name)]
	ylabel("Latitude", fontsize=14)
	xlabel("Longitude", fontsize=14)
	ylim(21.27, 21.33) #latitude
	xlim(-157.88, -157.8) #longitude
#	ylim(21.296, 21.2967)
#	xlim(-157.8198, -157.8189)

	
	#legend()
	title("Honolulu", fontsize=20)
	gcf()
#	savefig("Hono_trees.png", dpi=500)
#	"Honolulu/Oahu trees"
end

# ╔═╡ 087e57ab-aaea-48d6-b3ad-703d97d41b17
function overlap_area(tree_1, df)
	areaofoverlap = 0
	for eachtree in eachrow(df)
		if tree_1 != eachtree
			d = get_distance(tree_1, eachtree)
			if 0 < d < 2*R
				#meaning that there is some overlap but not complete
			#	θ = 2*acos(d/(2*R))
				#probably double check this equation
			#	A = R^2*(θ-sin(θ))
				A = 2*(R^2*acos(d/(2*R))-d/4*sqrt(4*R^2-d^2))
			elseif d == 0
				A = π*R^2
			elseif d >= 2*R
				A = 0
			end
			areaofoverlap += A
		end
		
	end
	return areaofoverlap
end

# ╔═╡ c4df8f67-ca37-4c12-bd01-5927c90a5fab
begin
	O_adulttrees[!, :num_overlaps] = [overlapping_trees(O_adulttrees[i, :], O_adulttrees) for i in 1:29]
	O_adulttrees[!, :area_overlaps] = [overlap_area(i, O_adulttrees) for i in eachrow(O_adulttrees)]
	O_adulttrees[!, :trees_in_circle] = [get_trees_in_circle(i, O_kidtrees) for i in eachrow(O_adulttrees)]

	O_kidtrees[!, :num_near_adults] = kid_trees_in_shared_space(O_kidtrees, O_adulttrees)
"Adding columns to O_adulttrees and O_kidtrees"
end

# ╔═╡ 4a3352b3-e602-4c60-a79b-73204d6abe3b
begin
	function bootstrap_t_test(data1, data2, n_boot = 1000)
		n1 = length(data1)
		n2 = length(data2)
		t_stat = n_boot
		ps = []
		for i in 1:n_boot
			sample1 = sample(data1, n1, replace= true)
			sample2 = sample(data2, n2, replace = true)
			test = MannWhitneyUTest(sample1, sample2)
			push!(ps, pvalue(test))
		end
		return ps
	end
end
			

# ╔═╡ f21bebb5-ed9d-467e-bfaf-c8933faa5c3f
pvalue(ExactMannWhitneyUTest(O_adulttrees.trees_in_circle, K_adulttrees.trees_in_circle))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
HypothesisTests = "09f84164-cd44-5f33-b23f-e6b0d136a0d5"
KernelDensity = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
PyPlot = "d330b81b-6aea-500a-939a-2ce795aea3ee"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ScikitLearn = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
Seaborn = "d2ef9438-c967-53ab-8060-373fdd9e13eb"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
CSV = "~0.10.4"
DataFrames = "~1.3.6"
HypothesisTests = "~0.11.4"
KernelDensity = "~0.6.10"
PyCall = "~1.96.4"
PyPlot = "~2.11.6"
ScikitLearn = "~0.6.4"
Seaborn = "~1.1.1"
StatsBase = "~0.33.21"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "051c95d6836228d120f5f4b984dd5aba1624f716"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "0.5.0"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[ArgTools]]
git-tree-sha1 = "bdf73eec6a88885256f282d48eafcad25d7de494"
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[Artifacts]]
deps = ["Pkg"]
git-tree-sha1 = "c30985d8821e0cd73870b17b0ed0ce6dc44cb744"
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.3.0"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings"]
git-tree-sha1 = "873fb188a4b9d76549b81465b1f75c82aaf59238"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.4"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "32ad4ece064a61855a35bdc34e3da0b495e01169"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.12.2"

[[ChangesOfVariables]]
deps = ["InverseFunctions", "LinearAlgebra", "Test"]
git-tree-sha1 = "3aa4bf1532aa2e14e0374c4fd72bed9a9d0d0f6c"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.10"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[Combinatorics]]
git-tree-sha1 = "c761b00e7755700f9cdf5b02039939d1359330e1"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.1.0"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "d476eaeddfcdf0de15a67a948331c69a585495fa"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.47.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8e695f735fca77e9708e795eda62afdb869cbb70"
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.3.4+0"

[[Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "8f06b0cfa4c514c7b9546756dbae91fcfbc92dc9"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.10.3"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "db2a9cb664fcea7836da4b414c3278d71dd602d2"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.6"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["AliasTables", "ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "0b4190661e8a4e51a842070e7dd4fae440ddb7f4"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.118"

[[DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
git-tree-sha1 = "39e99578597b4b1660b63cdabd5224ba53e3e71a"
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[FFTW]]
deps = ["AbstractFFTs", "BinaryProvider", "Conda", "Libdl", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "e0823a0ea2990b28a8398e958327333e8af53b27"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.1.1"

[[FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Test"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"

[[FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "deed294cde3de20ae0b2e0355a6c4e1c6a5ceffc"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.8"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[Formatting]]
deps = ["Logging", "Printf"]
git-tree-sha1 = "fb409abab2caf118986fc597ba84b50cbaf00b87"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.3"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[HypothesisTests]]
deps = ["Combinatorics", "Distributions", "LinearAlgebra", "Printf", "Random", "Rmath", "Roots", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68f07aa5e52f000da44d5160217a04fbb1d86a78"
uuid = "09f84164-cd44-5f33-b23f-e6b0d136a0d5"
version = "0.11.4"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "d19f9edd8c34760dca2de2b503f969d8700ed288"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.4"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[InverseFunctions]]
deps = ["Dates", "Test"]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"

[[InvertedIndices]]
git-tree-sha1 = "82aec7a3dd64f4d9584659dc0b62ef7db2ef3e19"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.2.0"

[[IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "ba51324b894edaf1df3ab16e2cc6bc3280a2f1a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.10"

[[LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
git-tree-sha1 = "7b8c8786a2d6913d0e873398166ecc4033d3fb9d"
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[LibCURL_jll]]
deps = ["LibSSH2_jll", "Libdl", "MbedTLS_jll", "Pkg", "Zlib_jll", "nghttp2_jll"]
git-tree-sha1 = "897d962c20031e6012bba7b3dcb7a667170dad17"
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.70.0+2"

[[LibGit2]]
deps = ["Printf"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Libdl", "MbedTLS_jll", "Pkg"]
git-tree-sha1 = "717705533148132e5466f2924b9a3657b16158e8"
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.9.0+3"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "b211c553c199c111d998ecdaf7623d1b89b69f93"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.12"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0eef589dd1c26a3ac9d753fe1a8bcad63f956fa6"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.16.8+1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4b835996a4a1fed59a8acdb1fd8b719dd932e2f8"
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2024.12.31+0"

[[NetworkOptions]]
git-tree-sha1 = "ed3157f48a05543cce9b241e1f2815f7e843d96e"
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"

[[OpenLibm_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "d22054f66695fe580009c09e765175cbf7f13031"
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.7.1+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9db77584158d0ab52307f8c04f8e7c08ca76b5b3"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.3+4"

[[OrderedCollections]]
git-tree-sha1 = "d78db6df34313deaca15c5c0b9ff562c704fe1ab"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.5.0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[Pandas]]
deps = ["Compat", "DataValues", "Dates", "IteratorInterfaceExtensions", "Lazy", "OrderedCollections", "Pkg", "PyCall", "Statistics", "TableTraits", "TableTraitsUtils", "Tables"]
git-tree-sha1 = "0ccb570180314e4dfa3ad81e49a3df97e1913dc2"
uuid = "eadc2687-ae89-51f9-a5d9-86b5a6373a9c"
version = "1.6.1"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "6c01a9b494f6d2a9fc180a08b182fcb06f0958a0"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.2"

[[Pkg]]
deps = ["Dates", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "UUIDs"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "9816a3826b0ebf49ab4926e2b18842ad8b5c8f04"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.96.4"

[[PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "d2c2b8627bbada1ba00af2951946fb8ce6012c05"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.11.6"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "e237232771fdafbae3db5c31275303e056afaa9f"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.10.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "86c5647b565873641538d8f812c04e4c9dbeb370"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.6.1"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "1b7bf41258f6c5c9c31df8c1ba34c1fc88674957"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.2.2+2"

[[Roots]]
deps = ["ChainRulesCore", "CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "0f1d92463a020321983d04c110f476c274bafe2e"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.0.22"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[ScikitLearn]]
deps = ["Compat", "Conda", "DataFrames", "Distributed", "IterTools", "LinearAlgebra", "MacroTools", "Parameters", "Printf", "PyCall", "Random", "ScikitLearnBase", "SparseArrays", "StatsBase", "VersionParsing"]
git-tree-sha1 = "ccb822ff4222fcf6ff43bbdbd7b80332690f168e"
uuid = "3646fa90-6ef7-5e7e-9f22-8aca16db6324"
version = "0.6.4"

[[ScikitLearnBase]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "7877e55c1523a4b336b433da39c8e8c08d2f221f"
uuid = "6e75b9c4-186b-50bd-896f-2d2496a4843e"
version = "0.5.0"

[[Seaborn]]
deps = ["Pandas", "PyCall", "PyPlot", "Reexport", "Test"]
git-tree-sha1 = "c7d0011bfb487a40501ad9383e24f1908809e1ed"
uuid = "d2ef9438-c967-53ab-8060-373fdd9e13eb"
version = "1.1.1"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "77172cadd2fdfa0c84c87e3a01215a4ca7723310"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.0.0"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "41852b8679f78c8d8961eeadc8f62cef861a52e3"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.1"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "74eaf352c0cef1e32ce7394bcc359d9199a28cf7"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.3.6"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "35b09e80be285516e52c9054792c884b9216ae3c"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.4.0"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
git-tree-sha1 = "44aaac2d2aec4a850302f9aa69127c74f0c3787e"
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[TableTraitsUtils]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Missings", "TableTraits"]
git-tree-sha1 = "78fecfe140d7abb480b53a44f3f85b6aa373c293"
uuid = "382cd787-c1b6-5bf2-a167-d5b971a19bda"
version = "1.0.2"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[Test]]
deps = ["Distributed", "InteractiveUtils", "Logging", "Random"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "9a6ae7ed916312b41236fcef7e0af564ef934769"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.13"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[Zlib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "320228915c8debb12cb434c59057290f0834dbf6"
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.11+18"

[[nghttp2_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "8e2c44ab4d49ad9518f359ed8b62f83ba8beede4"
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.40.0+2"
"""

# ╔═╡ Cell order:
# ╠═48706b3a-b446-11f0-263e-efa02e2f00be
# ╠═431bd934-b030-4aa2-8852-be426c0d0a6f
# ╟─b8d4de68-90a5-49fa-9587-699406b4d9d2
# ╟─026c47f9-a8f8-48b9-acf4-94365d27aa77
# ╠═c47ac6a7-38e0-45b3-b9bd-79a8ab79ab98
# ╠═364060ac-662d-47a6-8673-ac03bb65f321
# ╟─76c7835c-8386-4a45-9f9d-80a735f01a09
# ╟─73524a92-aecb-46b7-8a5a-8a099f6c4e3b
# ╠═c4df8f67-ca37-4c12-bd01-5927c90a5fab
# ╟─bbecb116-bf11-4d95-a78c-95fd6e0e2b39
# ╟─92bcaa09-0777-4365-b09f-82f5d137baba
# ╟─0e79c7fd-a60c-492e-9b9b-a1598f586959
# ╠═b513b08a-e041-4fa3-8e4d-21a2c79d9fda
# ╠═fcad27f6-c063-405b-a0b0-4730c0fead6c
# ╟─bf10ee1d-76e6-4a65-ae8d-df67ae6f724f
# ╟─5bcd0997-1fe7-4f5b-95b6-56e7a8822217
# ╠═091dcb1e-bf15-442a-8b0f-eaec5c11fde0
# ╠═862b5d5f-7a11-4219-9b2b-60c7d1fffb67
# ╠═cb6808ce-07b3-4fc7-ac76-fd7559082347
# ╠═71a30053-7f0a-415d-8dab-c4095fdb6485
# ╟─09adf665-2e8a-41d5-bf9c-db3b6a40a4ef
# ╟─ad2b5101-246a-4adf-ad52-c43685655b67
# ╟─52ff9693-f2c2-42e3-9211-02b66df20202
# ╟─32a4f1f2-cc4f-4c95-902f-f08056373046
# ╟─60cb902c-6648-4708-b070-12622d868fea
# ╟─3b3162b8-46c7-47da-88e3-e103197436ae
# ╟─5c147595-ec5d-4b82-8b52-e129e8ce75dc
# ╠═3c7b52dc-211d-485b-898b-94257cdc5951
# ╠═50fadefb-6cba-4a00-91ae-5d6eea604d0c
# ╟─c04ec5c0-fd1c-4eb5-a232-4dc9f04feb04
# ╟─5ea415db-4073-4b70-9180-332b7579ada0
# ╟─82c2344c-6a07-4187-bb30-6ac7918080cc
# ╟─2374a2a9-97ce-4bd7-8501-1c2fa4cb6b2e
# ╠═c28ead14-a09b-4641-aa45-a44488278248
# ╟─087e57ab-aaea-48d6-b3ad-703d97d41b17
# ╟─4a3352b3-e602-4c60-a79b-73204d6abe3b
# ╠═f21bebb5-ed9d-467e-bfaf-c8933faa5c3f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
