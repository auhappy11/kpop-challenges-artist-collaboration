#include <igraph/igraph.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Structure to store motif statistics
typedef struct {
    int motif_id;
    double original_count;
    double random_mean;
    double random_std;
    double z_score;
    double p_value;
    int is_significant;
    const char* description;
} MotifStats;

// Function to calculate Z-score and P-value
void calculate_statistics(MotifStats* stats, double* random_counts, int num_random) {
    // Calculate mean of random counts
    double sum = 0.0;
    for (int i = 0; i < num_random; i++) {
        sum += random_counts[i];
    }
    stats->random_mean = sum / num_random;
    
    // Calculate standard deviation
    double variance_sum = 0.0;
    for (int i = 0; i < num_random; i++) {
        double diff = random_counts[i] - stats->random_mean;
        variance_sum += diff * diff;
    }
    stats->random_std = sqrt(variance_sum / (num_random - 1));
    
    // Calculate Z-score only for motifs with actual instances
    if (stats->original_count == 0) {
        // Zero counts are not significant patterns
        stats->z_score = 0.0;
        stats->p_value = 1.0;
        stats->is_significant = 0;
    } else if (stats->random_std > 0.0001) { // Avoid division by zero
        stats->z_score = (stats->original_count - stats->random_mean) / stats->random_std;
        
        // Calculate precise P-value (two-tailed) using error function
        double abs_z = fabs(stats->z_score);
        // Standard Normal CDF: Φ(z) = 0.5 * (1 + erf(z/√2))
        double cdf = 0.5 * (1.0 + erf(abs_z / sqrt(2.0)));
        // Two-tailed P-value: P = 2 * (1 - Φ(|z|))
        stats->p_value = 2.0 * (1.0 - cdf);
        
        // Determine significance (Z-score > 1.0 is FANMOD standard threshold)
        stats->is_significant = (fabs(stats->z_score) >= 1.0);
    } else {
        stats->z_score = 0.0;
        stats->p_value = 1.0;
        stats->is_significant = 0;
    }
}

// Function to validate that degree sequence is preserved (both in- and out-degrees)
int validate_degree_sequence(const igraph_t* original, const igraph_t* randomized) {
    igraph_vector_int_t deg_orig_out, deg_rand_out, deg_orig_in, deg_rand_in;
    igraph_vector_int_init(&deg_orig_out, 0);
    igraph_vector_int_init(&deg_rand_out, 0);
    igraph_vector_int_init(&deg_orig_in, 0);
    igraph_vector_int_init(&deg_rand_in, 0);
    
    // Get out-degree sequences
    igraph_degree(original, &deg_orig_out, igraph_vss_all(), IGRAPH_OUT, IGRAPH_LOOPS);
    igraph_degree(randomized, &deg_rand_out, igraph_vss_all(), IGRAPH_OUT, IGRAPH_LOOPS);
    
    // Get in-degree sequences
    igraph_degree(original, &deg_orig_in, igraph_vss_all(), IGRAPH_IN, IGRAPH_LOOPS);
    igraph_degree(randomized, &deg_rand_in, igraph_vss_all(), IGRAPH_IN, IGRAPH_LOOPS);
    
    // Sort all degree sequences for comparison
    igraph_vector_int_sort(&deg_orig_out);
    igraph_vector_int_sort(&deg_rand_out);
    igraph_vector_int_sort(&deg_orig_in);
    igraph_vector_int_sort(&deg_rand_in);
    
    // Check if both in- and out-degree sequences are equal
    int out_equal = igraph_vector_int_all_e(&deg_orig_out, &deg_rand_out);
    int in_equal = igraph_vector_int_all_e(&deg_orig_in, &deg_rand_in);
    
    igraph_vector_int_destroy(&deg_orig_out);
    igraph_vector_int_destroy(&deg_rand_out);
    igraph_vector_int_destroy(&deg_orig_in);
    igraph_vector_int_destroy(&deg_rand_in);
    
    return (out_equal && in_equal);
}

// Improved function to perform edge switching for graph randomization
igraph_t create_random_graph(const igraph_t* original, int num_switches, int graph_id) {
    igraph_t random_graph;
    igraph_error_t err;
    
    // Copy the original graph
    err = igraph_copy(&random_graph, original);
    if (err != IGRAPH_SUCCESS) {
        printf("Error copying graph: %d\n", err);
        exit(1);
    }
    
    // Set igraph's random seed (this is crucial!)
    igraph_rng_seed(igraph_rng_default(), time(NULL) + graph_id);
    
    // Perform edge switching to preserve degree sequence
    // Use more switches for better randomization
    err = igraph_rewire(&random_graph, num_switches, IGRAPH_REWIRING_SIMPLE);
    if (err != IGRAPH_SUCCESS) {
        printf("Error rewiring graph %d: %d\n", graph_id, err);
        igraph_destroy(&random_graph);
        exit(1);
    }
    
    // Validate that degree sequence is preserved
    if (!validate_degree_sequence(original, &random_graph)) {
        printf("ERROR: Degree sequence not preserved in graph %d! Regenerating...\n", graph_id);
        igraph_destroy(&random_graph);
        
        // Try again with more aggressive rewiring
        err = igraph_copy(&random_graph, original);
        if (err != IGRAPH_SUCCESS) {
            printf("Error copying graph: %d\n", err);
            exit(1);
        }
        
        // Use different rewiring approach
        err = igraph_rewire(&random_graph, num_switches * 2, IGRAPH_REWIRING_SIMPLE);
        if (err != IGRAPH_SUCCESS) {
            printf("Error rewiring graph %d on retry: %d\n", graph_id, err);
            igraph_destroy(&random_graph);
            exit(1);
        }
        
        // Final check
        if (!validate_degree_sequence(original, &random_graph)) {
            printf("CRITICAL ERROR: Cannot preserve degree sequence for graph %d\n", graph_id);
            exit(1);
        } else {
            printf("Success: Degree sequence preserved on retry for graph %d\n", graph_id);
        }
    }
    
    return random_graph;
}

int main(int argc, char *argv[]) {
    igraph_t graph;
    igraph_vector_t hist_original, hist_random, cut_prob;
    igraph_vector_int_t edges;
    FILE *file;
    int from, to;
    char *input_file;
    char *output_prefix;
    
    // Check command line arguments
    if (argc != 3) {
        printf("Usage: %s <edge_file> <output_prefix>\n", argv[0]);
        printf("Example: %s motif_analysis/2022/edges_2022.txt 2022\n", argv[0]);
        printf("Example: %s motif_analysis/sliding_windows/2022-2023/edges_2022-2023.txt 2022-2023\n", argv[0]);
        return 1;
    }
    
    input_file = argv[1];
    output_prefix = argv[2];
    
    // Configuration parameters for FANMOD-style analysis
    const int NUM_RANDOM_GRAPHS = 5000;  // Number of random graphs to generate
    const double SIGNIFICANCE_THRESHOLD = 1.0; // Z-score threshold (FANMOD standard)
    
    printf("=== FANMOD-Compliant Motif Analysis ===\n");
    printf("Input file: %s\n", input_file);
    printf("Output prefix: %s\n", output_prefix);
    printf("Statistical significance testing with %d random graphs\n", NUM_RANDOM_GRAPHS);
    
    // Initialize igraph's random number generator
    igraph_rng_seed(igraph_rng_default(), time(NULL));
    
    // Load graph from specified edge list file
    printf("Loading artist collaboration graph from edge list...\n");
    
    // Initialize edge vector
    igraph_vector_int_init(&edges, 0);
    
    // Read edges from specified file
    file = fopen(input_file, "r");
    if (!file) {
        printf("Error: Cannot open edge list file '%s'\n", input_file);
        printf("Please make sure the file exists and has correct format.\n");
        return 1;
    }
    
    while (fscanf(file, "%d %d", &from, &to) == 2) {
        igraph_vector_int_push_back(&edges, from);
        igraph_vector_int_push_back(&edges, to);
    }
    fclose(file);
    
    printf("Read %lld edges\n", (long long)igraph_vector_int_size(&edges) / 2);
    
    // Create graph
    igraph_create(&graph, &edges, 207, IGRAPH_DIRECTED);
    
    printf("Graph loaded: %lld vertices, %lld edges\n", 
           (long long)igraph_vcount(&graph), (long long)igraph_ecount(&graph));
    
    // Calculate FANMOD-standard edge switches: 10x number of edges
    const int NUM_EDGE_SWITCHES = 10 * igraph_ecount(&graph);
    printf("Edge switches per randomization: %d (10x edges)\n", NUM_EDGE_SWITCHES);
    
    // Initialize result vectors
    igraph_vector_init(&hist_original, 0);
    igraph_vector_init(&hist_random, 0);
    igraph_vector_init(&cut_prob, 0);
    
    // === 3-node motif analysis with statistical testing ===
    printf("\n=== 3-node motif analysis with statistical significance ===\n");
    
    // Initialize cut_prob for 3-node motifs (complete enumeration)
    igraph_vector_resize(&cut_prob, 3);
    VECTOR(cut_prob)[0] = 0.0;
    VECTOR(cut_prob)[1] = 0.0;
    VECTOR(cut_prob)[2] = 0.0;
    
    // Count motifs in original graph
    printf("Analyzing original network...\n");
    igraph_motifs_randesu(&graph, &hist_original, 3, &cut_prob);
    
    // Get the size of motif histogram
    int num_motif_types = igraph_vector_size(&hist_original);
    
    // Arrays to store random graph results
    double** random_results = malloc(num_motif_types * sizeof(double*));
    for (int i = 0; i < num_motif_types; i++) {
        random_results[i] = malloc(NUM_RANDOM_GRAPHS * sizeof(double));
    }
    
    // Degree preservation tracking
    int degree_preserved_count = 0;
    
    // Generate random graphs and count motifs
    printf("Generating and analyzing %d random graphs (this may take a while)...\n", NUM_RANDOM_GRAPHS);
    
    for (int rand_i = 0; rand_i < NUM_RANDOM_GRAPHS; rand_i++) {
        if ((rand_i + 1) % 50 == 0) {
            printf("Progress: %d/%d random graphs completed\n", rand_i + 1, NUM_RANDOM_GRAPHS);
        }
        
        // Create randomized graph with proper seeding
        igraph_t random_graph = create_random_graph(&graph, NUM_EDGE_SWITCHES, rand_i);
        
        // Track degree preservation success
        if (validate_degree_sequence(&graph, &random_graph)) {
            degree_preserved_count++;
        }
        
        // Count motifs in random graph
        igraph_motifs_randesu(&random_graph, &hist_random, 3, &cut_prob);
        
        // Store results
        for (int motif_i = 0; motif_i < num_motif_types; motif_i++) {
            if (motif_i < igraph_vector_size(&hist_random)) {
                random_results[motif_i][rand_i] = VECTOR(hist_random)[motif_i];
            } else {
                random_results[motif_i][rand_i] = 0.0;
            }
        }
        
        // Clean up random graph
        igraph_destroy(&random_graph);
    }
    
    // No motif descriptions - output raw statistics only
    
    // Calculate statistics and identify significant motifs
    printf("\n=== Statistical Analysis Results ===\n");
    printf("Motif ID | Original | Random Mean | Std Dev | Z-Score | P-Value | Significant\n");
    printf("---------|----------|-------------|---------|---------|---------|-------------\n");
    
    int significant_motifs_count = 0;
    MotifStats* motif_stats = malloc(num_motif_types * sizeof(MotifStats));
    
    for (int i = 0; i < num_motif_types; i++) {
        // igraph reports the disconnected 3-node classes (0, 1, 3) as NaN; skip
        // them so only the 13 connected motifs are analysed and reported.
        if (!isnan(VECTOR(hist_original)[i])) {
            motif_stats[i].motif_id = i;
            motif_stats[i].original_count = VECTOR(hist_original)[i];
            
            calculate_statistics(&motif_stats[i], random_results[i], NUM_RANDOM_GRAPHS);
            
            printf("%8d | %8.0f | %11.2f | %7.2f | %7.3f | %7.3f | %11s\n",
                   motif_stats[i].motif_id,
                   motif_stats[i].original_count,
                   motif_stats[i].random_mean,
                   motif_stats[i].random_std,
                   motif_stats[i].z_score,
                   motif_stats[i].p_value,
                   motif_stats[i].is_significant ? "YES" : "NO");
            
            if (motif_stats[i].is_significant) {
                significant_motifs_count++;
            }
        }
    }
    
    printf("\n=== Summary ===\n");
    printf("Total significant 3-node motifs found: %d\n", significant_motifs_count);
    printf("Significance threshold: Z-score >= %.1f\n", SIGNIFICANCE_THRESHOLD);
    printf("Degree sequence preserved in %d/%d random graphs (%.1f%%)\n", 
           degree_preserved_count, NUM_RANDOM_GRAPHS, 
           100.0 * degree_preserved_count / NUM_RANDOM_GRAPHS);
    
    // Save detailed results
    char output_filename[256];
    snprintf(output_filename, sizeof(output_filename), "fanmod_motif_analysis_%s.txt", output_prefix);
    printf("\nSaving detailed results to '%s'...\n", output_filename);
    FILE *output = fopen(output_filename, "w");
    if (output) {
        fprintf(output, "FANMOD-Style Motif Analysis Results\n");
        fprintf(output, "===================================\n");
        fprintf(output, "Network: Artist Collaboration Graph (%s)\n", output_prefix);
        fprintf(output, "Input file: %s\n", input_file);
        fprintf(output, "Vertices: %lld\n", (long long)igraph_vcount(&graph));
        fprintf(output, "Edges: %lld\n", (long long)igraph_ecount(&graph));
        fprintf(output, "Random graphs generated: %d\n", NUM_RANDOM_GRAPHS);
        fprintf(output, "Edge switches per randomization: %d\n", NUM_EDGE_SWITCHES);
        fprintf(output, "Significance threshold: Z >= %.1f\n\n", SIGNIFICANCE_THRESHOLD);
        
        fprintf(output, "RAW MOTIF STATISTICS:\n");
        fprintf(output, "====================\n");
        fprintf(output, "MotifID\tOriginal\tRandomMean\tRandomStd\tZScore\tPValue\tSignificant\n");
        
        int sig_count = 0;
        for (int i = 0; i < num_motif_types; i++) {
            if (!isnan(VECTOR(hist_original)[i])) { // same 13 connected motifs as above
                fprintf(output, "%d\t%.0f\t%.2f\t%.2f\t%.3f\t%.3f\t%s\n", 
                       i, 
                       motif_stats[i].original_count,
                       motif_stats[i].random_mean, 
                       motif_stats[i].random_std,
                       motif_stats[i].z_score,
                       motif_stats[i].p_value,
                       motif_stats[i].is_significant ? "YES" : "NO");
                
                if (motif_stats[i].is_significant) {
                    sig_count++;
                }
            }
        }
        
        fprintf(output, "Total significant motifs: %d\n", sig_count);
        
        fclose(output);
        printf("Results saved successfully!\n");
    }
    
    // Clean up memory
    for (int i = 0; i < num_motif_types; i++) {
        free(random_results[i]);
    }
    free(random_results);
    free(motif_stats);
    
    // Clean up igraph objects
    igraph_vector_int_destroy(&edges);
    igraph_vector_destroy(&hist_original);
    igraph_vector_destroy(&hist_random);
    igraph_vector_destroy(&cut_prob);
    igraph_destroy(&graph);
    
    printf("\nFANMOD-compliant motif analysis completed!\n");
    printf("True motifs are those with statistical significance (Z >= %.1f)\n", SIGNIFICANCE_THRESHOLD);
    
    return 0;
}