# Dermatify Model Loader (Internal Service)

## Endpoints

## Register

1. `POST` to endpoint `/predict` with an image:

**Response**

```
{
    "issue": "Acne",
    "score": 0.8
}
```

**Error Responses**

```
{
    "message": "Model note loaded yet."
}
```