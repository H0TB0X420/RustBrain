#[cfg(test)]
mod tests {
    use rustbrain::math::Matrix;

#[test]
fn test_lu_decomposition_triangular_forms() {
    let a = Matrix::new(vec![
        vec![4.0, 3.0, 2.0],
        vec![2.0, 1.0, 3.0],
        vec![3.0, 2.0, 1.0],
    ]);

    let (lu, _) = a.lu_decomposition();
    let (l, u) = lu.split_lu(); // Assuming split_lu() extracts L and U separately

    // Check if L is lower triangular
    for i in 0..l.row_count() {
        for j in (i + 1)..l.col_count() {
            assert!(
                (l[(i, j)] - 0.0).abs() < 1e-6,
                "L is not lower triangular at ({}, {}): found {}",
                i, j, l[(i, j)]
            );
        }
    }
    println!("{}", l);
    println!("{}", u);
    // Check if U is upper triangular
    for i in 1..u.row_count() {
        for j in 0..i {
            assert!(
                (u[(i, j)] - 0.0).abs() < 1e-6,
                "U is not upper triangular at ({}, {}): found {}",
                i, j, u[(i, j)]
            );
        }
    }
}
}
