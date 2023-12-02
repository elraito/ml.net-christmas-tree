using System.Text.Json.Serialization;
using Microsoft.ML.Data;

namespace Server.Models;

public class ModelOutput 
{
    [JsonIgnore]
    public string ImagePath { get; set; } = default!;
    public string Label { get; set; } = default!;
    public string PredictedLabel { get; set; } = default!;
    [ColumnName("Score")]
    public float[] Probability { get; set; } = default!;
}