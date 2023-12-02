using Microsoft.AspNetCore.Mvc;
using Server.Services;

namespace Server;

public static class ModelBuilderRoutes
{
    public static void MapModelBuilderRoutes(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapGet("/train", (IModelBuilderService _modelBuilder) => 
        {
            _modelBuilder.TrainModel();
            return Results.Ok("Building model...");
        })
        .WithName("Trainodel")
        .WithOpenApi();

        endpoints.MapPost("/predict", async(IInferenceService _inferenceService, [FromForm] PredictDto? predictDto) => 
        {
            if (predictDto == null || predictDto.File == null || predictDto.File.Length == 0)
                return Results.BadRequest("No file");

            byte [] fileData;
            
            using var memoryStream = new MemoryStream();
            await predictDto.File.CopyToAsync(memoryStream);
            fileData = memoryStream.ToArray();

            var result = _inferenceService.Predict(fileData, predictDto.File.FileName);

            return Results.Ok(result);
        })
        .WithName("Predict")
        .WithOpenApi()
        .DisableAntiforgery();
    }
}

public class PredictDto
{
    public IFormFile File { get; set; } = default!;
}