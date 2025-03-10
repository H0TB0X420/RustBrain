#[cfg(test)]
mod tests {
    use rustbrain::math::Vector;

    #[test]
    fn test_vector_creation() {
        let v = Vector::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_vector_dot_product() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let result = v1.dot(&v2);
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_vector_addition() {
        let v1 = Vector::new(vec![1.0, 2.0, 3.0]);
        let v2 = Vector::new(vec![4.0, 5.0, 6.0]);
        let result = v1.add(&v2);
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    }
}
